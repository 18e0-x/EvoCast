## 1. 安装环境

创建 Python 环境并安装依赖：

```shell
conda create -n char_extractor python=3.10 -y
conda activate char_extractor
pip install pandas scipy numpy statsmodels scikit-learn
```

## 2. 输入

提取器支持宽表 CSV，也支持 TFB 三列长表格式。

- 单个文件：`./DemoDatasets/Exchange.csv`
- 文件夹：`./DemoDatasets`

文件夹模式会处理该目录下一层的所有 CSV 文件。

## 3. 输出

对于单变量文件，例如 `m4_hourly_dataset_69.csv`，输出目录包含：

- `All_characteristics_m4_hourly_dataset_69.csv`
- `TFB_characteristics_m4_hourly_dataset_69.csv`

对于多变量文件，例如 `Exchange.csv`，输出目录包含：

- `All_characteristics_Exchange.csv`
- `TFB_characteristics_Exchange.csv`
- `mean_All_characteristics_Exchange.csv`
- `mean_TFB_characteristics_Exchange.csv`

`TFB_characteristics_*.csv` 和 `mean_TFB_characteristics_*.csv` 包含以下列：

- `Correlation`
- `Transition`
- `Shifting`
- `Seasonality`
- `Trend`
- `Stationarity`
- `Short_term_jsd`
- `Long_term_jsd`

## 4. 特征定义

特征定义见 `docs/tutorials/introduction_and_pseudocode_for_time_series_characteristics.md`。

## 5. 代码文件

实现文件是 `characteristics_extractor/Characteristics_Extractor.py`。
