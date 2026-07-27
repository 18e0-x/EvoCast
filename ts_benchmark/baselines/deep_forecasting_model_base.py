import copy
import math
import os
import re
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _extract_base_freq(freq_str: str) -> str:
    """Extract base frequency unit from pandas offset string.

    '15T' → 't', '30min' → 't', 'h' → 'h', 'D' → 'd', 'MS' → 'm'.
    """
    if not freq_str:
        return "h"
    m = re.match(r"\d*([a-zA-Z]+)", freq_str)
    if not m:
        return "h"
    base = m.group(1).lower()
    return {"min": "t", "minute": "t", "hour": "h", "day": "d", "month": "m",
            "week": "w", "second": "s", "sec": "s", "business": "b"}.get(base, base[:1])


_FREQ_DATE_RANGE = {"h": "h", "t": "min", "s": "s", "d": "D", "w": "W", "m": "MS", "b": "B"}
import logging
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from ts_benchmark.utils.get_device import get_device

from ts_benchmark.baselines.utils import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)
from ts_benchmark.models.model_base import ModelBase, BatchMaker
from ts_benchmark.utils.data_processing import split_time

logger = logging.getLogger(__name__)

# Default hyper parameters
DEFAULT_HYPER_PARAMS = {
    "use_amp": 0,
    "loss": "MSE",
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.0001,
    "num_workers": 0,
    "patience": 10,
    "num_epochs": 100,
    "adj_lr_in_epoch": True,
    "adj_lr_in_batch": False,
    "parallel_strategy": None,
}


class Config:
    def __init__(self, model_config, **kwargs):

        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in model_config.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Backward compatibility for configs that used the pluralized
        # TimeMixer field name. The model implementations read the
        # singular form `down_sampling_window`.
        if hasattr(self, "down_sampling_windows") and not hasattr(self, "down_sampling_window"):
            setattr(self, "down_sampling_window", self.down_sampling_windows)

        if hasattr(self, "horizon"):
            import warnings
            warnings.warn(
                "The model parameter horizon is deprecated. Please use pred_len.",
                FutureWarning, stacklevel=2,
            )
            setattr(self, "pred_len", self.horizon)


class DeepForecastingModelBase(ModelBase):
    """
    Base class for deep learning model in forecasting tasks, inherited from ModelBase.

    This class provides a framework and default functionalities for adapters in time series forecasting tasks,
    including model initialization, configuration of loss functions and optimizers, data processing,
    learning rate adjustment, save checkpoints and early stopping mechanisms.

    Subclasses must implement _init_model and _process methods to define specific data processing and modeling logic.

    """

    def __init__(self, model_config, **kwargs):
        super(DeepForecastingModelBase, self).__init__()
        self.config = Config(model_config, **kwargs)
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len
        self.check_point = None

    def _init_model(self):
        """
        Initialize the model.

        This method is intended to be implemented by subclasses to initialize the specific model.
        The current implementation raises a NotImplementedError to indicate that this method should
        be overridden in subclasses.

        :return: The actual model object. The specific type of the return value should be defined by subclasses.
        """
        raise NotImplementedError("model must be implemented.")

    def _adjust_lr(self, optimizer, epoch, config):
        """
        Adjusts the learning rate of the optimizer based on the current epoch and configuration.

        This method is typically called to update the learning rate according to a predefined schedule.

        :param optimizer: The optimizer for which the learning rate will be adjusted.
        :param epoch: The current training epoch used to calculate the new learning rate.
        :param config: Configuration object containing parameters that control learning rate adjustment.
        """
        adjust_learning_rate(optimizer, epoch, config)

    def save_checkpoint(self, model):
        """
        Save the model checkpoint.

        This function saves the model's state dictionary (state_dict) to be used
        for restoring the model at a later time. A deep copy of the state_dict is returned.

        Parameters:
        - model (torch.nn.Module): The current instance of the model being trained.

        Returns:
        - OrderedDict: A deep copy of the model's state_dict, which can be used to restore
          the model's parameters in the future.
        """
        return copy.deepcopy(model.state_dict())

    def _init_criterion_and_optimizer(self):
        """
        Initializes the task loss function and optimizer.

        This method configures the task loss function and the optimizer based on the settings in `self.config`.
        Default supported loss functions include Mean Squared Error (MSE), Mean Absolute Error (MAE), and Huber Loss.
        And the Adam optimizer is used with the model's parameters and the learning rate specified in the configuration.

        :return: A tuple containing the initialized task loss function (`criterion`) and the optimizer (`optimizer`).
        """
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _process(self, input, target, input_mark, target_mark):
        """
        A method that needs to be implemented by subclasses to process data and model, and calculate additional loss.

        This method's purpose is to serve as a template method, defining a standard process for data processing
        and modeling, as well as calculating any additional losses. Subclasses should implement specific processing
        and calculation logic based on their own needs.

        Parameters:
        - input: The input data, the specific form and meaning depend on the implementation of the subclass.
        - target: The target data, used in conjunction with input data for processing and loss calculation.
        - input_mark: Marks or metadata for the input data, assisting in data processing or model training.
        - target_mark: Marks or metadata for the target data, similarly assisting in data processing or model training.

        Returns:
        - dict: A dictionary containing at least one key:
            - 'output' (necessary): The model output tensor.
            - 'additional_loss' (optional): An additional loss if it exists.

        Raises:
        - NotImplementedError: If the subclass does not implement this method, a NotImplementedError will be raised
                               when calling this method.
        """
        raise NotImplementedError("Process must be implemented")

    def _post_process(self, output, target):
        """
        Performs post-processing on the output and target data.

        This function is designed to process the output and target data after the model's forward computation,
        and return them directly in this example. The specific post-processing logic may include, but is not limited to,
        data format conversion, dimensionality matching, data type conversion, etc.

        Parameters:
        - output: The output data from the model, with no specific data format or type assumed.
        - target: The target data, which is the expected result, also without a fixed data format or type.

        Returns:
        - output: The output data after post-processing, which in this case is the same as the input.
        - target: The target data after post-processing, which in this case is the same as the input.
        """
        return output, target

    def _init_early_stopping(self):
        """
        Initializes the early stopping strategy for training.

        This function is used to create an instance of EarlyStopping, which helps prevent overfitting
        during model training by halting the training process when the validation performance
        does not improve for a specified number of consecutive iterations.

        Parameters:
        None directly, but it uses self.config.patience as the patience parameter for EarlyStopping.

        Returns:
        An instance of EarlyStopping, which monitors the model's performance metrics and determines
        when to stop the training.
        """
        return EarlyStopping(patience=self.config.patience)

    @property
    def model_name(self):
        return "DeepForecastingModelBase"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        base = _extract_base_freq(freq)
        self.config.freq = base
        self.config.pandas_freq = _FREQ_DATE_RANGE.get(base, base)

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        base = _extract_base_freq(freq)
        self.config.freq = base
        self.config.pandas_freq = _FREQ_DATE_RANGE.get(base, base)

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.horizon)

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        base = _extract_base_freq(freq)
        self.config.freq = base
        self.config.pandas_freq = _FREQ_DATE_RANGE.get(base, base)

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.horizon + 1)
        date = pd.date_range(
            start=start, periods=self.config.horizon + 1,
            freq=_FREQ_DATE_RANGE.get(self.config.freq, self.config.freq),
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        """
        Padding time stamp mark for prediction.

        :param time_stamps_list: A batch of time stamps.
        :param padding_len: The len of time stamp need to be padded.
        :return: The padded time stamp mark.
        """
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=_FREQ_DATE_RANGE.get(self.config.freq, self.config.freq),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(
            whole_time_stamp, 1,
            _FREQ_DATE_RANGE.get(self.config.freq, self.config.freq),
        )
        return padding_mark

    def validate(
        self, valid_data_loader: DataLoader, series_dim: int, criterion: torch.nn.Module
    ) -> float:
        """
        Validates the model performance on the provided validation dataset.
        :param valid_data_loader: A PyTorch DataLoader for the validation dataset.
        :param series_dim : The number of series data’s dimensions.
        :param criterion : The loss function to compute the loss between model predictions and ground truth.
        :returns:The mean loss computed over the validation dataset.
        """
        config = self.config
        max_val_batches = int(getattr(config, "max_val_batches", 0) or 0)
        total_loss = []
        self.model.eval()
        device = get_device()
        val_batches = 0
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )

                out_loss = self._process(input, target, input_mark, target_mark)
                additional_loss = 0
                output = out_loss["output"]
                if "additional_loss" in out_loss:
                    additional_loss = out_loss["additional_loss"]
                target = target[:, -config.horizon :, :series_dim]
                output = output[:, -config.horizon :, :series_dim]
                output, target = self._post_process(output, target)
                all_loss = criterion(output, target) + additional_loss
                loss = all_loss.detach().cpu().numpy()
                total_loss.append(loss)
                val_batches += 1
                if max_val_batches > 0 and val_batches >= max_val_batches:
                    break

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "ModelBase":
        """
        Train the model.
        :param train_valid_data: Time series data used for training and validation.
        :param covariates: Additional external variables.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        if covariates is None:
            covariates = {}
        series_dim = train_valid_data.shape[-1]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            train_valid_data = pd.concat([train_valid_data, exog_data], axis=1)

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        self.model = self._init_model()

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        if os.environ.get("TFB_VERBOSE"):
            print(
                "----------------------------------------------------------",
                self.model_name,
            )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        self.scaler.fit(train_data.values)

        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        train_dataset, self.train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )

        # Define the loss function and optimizer
        criterion, optimizer = self._init_criterion_and_optimizer()

        if config.use_amp == 1:
            scaler = torch.cuda.amp.GradScaler()


        device = get_device()


        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        if os.environ.get("TFB_VERBOSE"):
            print(f"Total trainable parameters: {total_params}")

        train_started_at = time.time()
        num_epochs = int(config.num_epochs)
        max_train_batches = int(getattr(config, "max_train_batches", 0) or 0)
        for epoch in range(num_epochs):
            epoch_started_at = time.time()
            epoch_loss_sum = 0.0
            epoch_batches = 0
            self.model.train()
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                self.train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input
                out_loss = self._process(input, target, input_mark, target_mark)
                additional_loss = 0
                output = out_loss["output"]
                if "additional_loss" in out_loss:
                    additional_loss = out_loss["additional_loss"]

                target = target[:, -config.horizon :, :series_dim]
                output = output[:, -config.horizon :, :series_dim]
                output, target = self._post_process(output, target)
                loss = criterion(output, target)

                total_loss = loss + additional_loss

                if config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                epoch_loss_sum += float(total_loss.detach().cpu().item())
                epoch_batches += 1

                if self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

                if max_train_batches > 0 and epoch_batches >= max_train_batches:
                    break

            valid_loss = None
            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                improved = self.early_stopping(valid_loss, self.model)
                # P2-build: skip checkpoint save in smoke/build mode (1 epoch).
                # num_epochs == 1 with max_train_batches means this is a smoke run;
                # saving the model checkpoint is wasted GPU→CPU transfer + disk I/O.
                max_train = int(getattr(config, "max_train_batches", 0) or 0)
                if improved and not (int(config.num_epochs) <= 1 and max_train > 0):
                    self.check_point = self.save_checkpoint(self.model)

            if self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)

            elapsed = time.time() - train_started_at
            epoch_elapsed = time.time() - epoch_started_at
            avg_epoch = elapsed / max(epoch + 1, 1)
            eta_seconds = max(num_epochs - epoch - 1, 0) * avg_epoch
            train_loss = epoch_loss_sum / max(epoch_batches, 1)
            msg = (
                f"[epoch] {self.model_name} "
                f"{epoch + 1}/{num_epochs} "
                f"train_loss={train_loss:.6f}"
            )
            if valid_loss is not None:
                msg += f" val_loss={valid_loss:.6f}"
            msg += (
                f" epoch_time={epoch_elapsed:.1f}s "
                f"elapsed={elapsed:.1f}s "
                f"eta={eta_seconds:.1f}s"
            )
            print(msg)

            if train_ratio_in_tv != 1 and self.early_stopping.early_stop:
                print(f"[epoch] {self.model_name} early stopped at {epoch + 1}/{num_epochs}")
                break

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Make predictions.
        :param horizon: The predicted length.
        :param series: Time series data used for prediction.
        :param covariates: Additional external variables
        :return: An array of predicted results.
        """
        if covariates is None:
            covariates = {}
        series_dim = series.shape[-1]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            series = pd.concat([series, exog_data], axis=1)
            if (
                hasattr(self.config, "output_chunk_length")
                and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )

        if self.check_point is not None:
            self.model.load_state_dict(self.check_point)

        if self.config.norm:
            series = pd.DataFrame(
                self.scaler.transform(series.values),
                columns=series.columns,
                index=series.index,
            )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        series, test = split_time(series, len(series) - config.seq_len)
        test = self.padding_data_for_forecast(test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        device = get_device()
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            answer = None
            while answer is None or answer.shape[0] < horizon:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )

                    out_loss = self._process(input, target, input_mark, target_mark)
                    output = out_loss["output"]

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon :]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    if self.config.norm:
                        answer[-horizon:] = self.scaler.inverse_transform(
                            answer[-horizon:]
                        )
                    return answer[-horizon:, :series_dim]

                output = output.cpu().numpy()[:, -config.horizon :]
                for i in range(config.horizon):
                    test.iloc[i + config.seq_len] = output[0, i, :]

                test = test.iloc[config.horizon :]
                test = self.padding_data_for_forecast(test)

                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :return: An array of predicted results.
        """
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = get_device()
        self.model.to(device)
        self.model.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]

        series_dim = input_np.shape[-1]

        if input_data["covariates"] is None:
            covariates = {}
        else:
            covariates = input_data["covariates"]
        exog_data = covariates.get("exog")
        if exog_data is not None:
            input_np = np.concatenate((input_np, exog_data), axis=2)
            if (
                hasattr(self.config, "output_chunk_length")
                and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )
        if self.config.norm:
            origin_shape = input_np.shape
            flattened_data = input_np.reshape((-1, input_np.shape[-1]))
            input_np = self.scaler.transform(flattened_data).reshape(origin_shape)

        input_index = input_data["time_stamps"]
        padding_len = (
            math.ceil(horizon / self.config.horizon) + 1
        ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)

        answers = self._perform_rolling_predictions(horizon, input_np, all_mark, device)

        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers[..., :series_dim]

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        """
        Perform rolling predictions using the given input data and marks.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param device: Device to run the model on.
        :return: List of predicted results for each prediction batch.
        """
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )

                out_loss = self._process(input, dec_input, input_mark, target_mark)
                output = out_loss["output"]
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                        :, -self.config.horizon :, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon :, :]
                (
                    input_np,
                    target_np,
                    input_mark_np,
                    target_mark_np,
                ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare rolling data based on the current rolling time.

        :param input_np: Current input data.
        :param output: Output from the model prediction.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param rolling_time: Current rolling time step.
        :return: Updated input data, target data, input marks, and target marks for rolling prediction.
        """
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len :, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
            :, -self.config.label_len :, :
        ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len : self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[
            :,
            start:end,
            :,
        ]
        return input_np, target_np, input_mark_np, target_mark_np
