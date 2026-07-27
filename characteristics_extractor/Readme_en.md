## 1. Environment Setup

Create a Python environment and install the required Python packages:

```shell
conda create -n char_extractor python=3.10 -y
conda activate char_extractor
pip install pandas scipy numpy statsmodels scikit-learn
```

## 2. Input

The extractor accepts CSV time-series data in either wide-table format or TFB three-column long-table format.

- Single file: `./DemoDatasets/Exchange.csv`
- Directory: `./DemoDatasets`

Directory mode processes every CSV file directly under the given directory.

## 3. Output

For a univariate file such as `m4_hourly_dataset_69.csv`, the output directory contains:

- `All_characteristics_m4_hourly_dataset_69.csv`
- `TFB_characteristics_m4_hourly_dataset_69.csv`

For a multivariate file such as `Exchange.csv`, the output directory contains:

- `All_characteristics_Exchange.csv`
- `TFB_characteristics_Exchange.csv`
- `mean_All_characteristics_Exchange.csv`
- `mean_TFB_characteristics_Exchange.csv`

`TFB_characteristics_*.csv` and `mean_TFB_characteristics_*.csv` contain these columns:

- `Correlation`
- `Transition`
- `Shifting`
- `Seasonality`
- `Trend`
- `Stationarity`
- `Short_term_jsd`
- `Long_term_jsd`

## 4. Feature Definitions

For the characteristic definitions, see `docs/tutorials/introduction_and_pseudocode_for_time_series_characteristics.md`.

## 5. Code File

The implementation is `characteristics_extractor/Characteristics_Extractor.py`.
