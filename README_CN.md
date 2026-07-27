<div align="center">
  <img src="assets/icon.png" width="50%" alt="EvoCast icon">
  <h1>EvoCast：面向迭代式预测架构演化的可靠自主研究 Agent</h1>
  <p>
    <a href="./README.md"><img src="https://img.shields.io/badge/English-Switch-2f81f7?style=flat-square" alt="English"></a>
    <a href="./README_CN.md"><img src="https://img.shields.io/badge/简体中文-当前-8b5cf6?style=flat-square" alt="简体中文"></a>
  </p>
  <p><em>“只剩细胞核了可以用 EvoCast 开发时序预测模型吗？”</em></p>
  <p>EvoCast 是一个面向深度学习时序预测开发的自主研究系统，能够在真实代码仓库中自动执行模块级架构组合、功能插入、候选实现与实验验证。它围绕给定数据集持续改进选定的模型结构，为不同领域构建任务特定的专用时序预测模型。所有研究过程都在统一、可审计的评测协议下展开，从而保证开发路径透明、结果比较公平、实验记录完整。</p>
</div>

<p align="center">
  <img src="assets/evocast_system_overview.png" alt="EvoCast overview" width="940">
</p>

---

## Workflow Overview 🔁

EvoCast 将 forecasting architecture development 组织为固定 research loop。给定 forecasting task 和源码仓库后，系统首先建立任务特定 baseline，并收集任务证据与机制证据。随后，系统执行受控 research rounds，生成受边界约束的源码候选，验证修改内容，在 canonical forecasting protocol 下完成评测，并持续更新持久化研究状态。整个流程最终生成可审计的 HTML 报告，用于汇总 baseline results、candidate attempts、metrics、decisions 和 round history。

## 仓库结构 🗂️

```text
./
  evocast/                    EvoCast 主体代码：任务、研究、构建、评测与报告
  ts_benchmark/               TFB/ts_benchmark 集成与 forecasting backbones
  characteristics_extractor/  数据集分析工具
  config/                     forecasting protocol 配置
  tests/                      单元测试与回归测试
  assets/                     README 图片与静态资源
```

## Dataset Placement 📊

EvoCast 支持两种数据集输入方式：直接传入显式 CSV 路径，或将数据集放在仓库的 `dataset/` 目录下。

推荐目录结构：

```text
./
  dataset/
    forecasting/
      ETTh1.csv
      ETTh2.csv
      ETTm1.csv
    custom_task.csv
```

数据集路径按以下顺序解析：

1. `--dataset` 传入的精确路径
2. `dataset/forecasting/<filename>`
3. `dataset/<filename>`

CSV 要求：

- 文件必须是可读取的 CSV。
- 必须存在时间列。常见列名包括 `date`、`time`、`timestamp`、`datetime` 和 `ds`。
- 宽表格式 forecasting CSV 应包含一列时间列和一列或多列数值特征列。
- 也支持 `date/time`、`data`、`cols` 结构的长表格式 CSV。

任务模式要求：

- `MM`：多变量输入并预测所有目标变量；不要求显式指定目标列。
- `MS`：多变量输入并预测唯一目标列；请通过 `--target` 指定。
- `SS`：单变量输入并预测一个目标列；建议显式传入 `--target`。

实际行为说明：

- 交互模式下如果没有提供 `--time-col`，向导会明确要求用户选择时间列。
- 如果目标列与所选任务模式不一致，校验器会在执行前直接报错。
- 宽表中的非数值列会在建模前被剔除，不能作为 forecasting target 使用。

最小示例：

```bash
python -m evocast --dataset dataset/forecasting/ETTh1.csv --time-col date
```

## 环境 ⚙️

在仓库根目录创建并激活干净的 Python 环境：

```bash
git clone <your-repo-url>
cd <repository-root>
python -m venv .venv
```

激活环境：

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

## Requirements 📋

- Python `>=3.10`
- 兼容 PyTorch 的运行环境
- 本地 forecasting 数据集 CSV
- 运行 LLM research workflow 所需的 provider API key

`requirements.txt` 安装 EvoCast 的快速启动环境。`requirements-full.txt` 安装仓库完整环境，包含 dashboard、Darts、Merlion、Mamba、并行执行和开发工具。

## Installation 📦

安装快速启动环境：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

需要完整仓库环境时，执行：

```bash
python -m pip install -r requirements-full.txt
```

## Default Parameters 🎛️

以下默认项对应当前交互式向导和运行时行为。

| 组件 | 设置 |
| --- | --- |
| 启动入口 | `python -m evocast` |
| 任务定义 | dataset、mode、target、lookback、horizon、objective |
| Baseline 设置 | 自动搜索或手动指定 |
| 数据集分析 | `required`、`reuse` 或 `skip` |
| Research rounds | 受控迭代轮次与有边界的源码修改 |
| 输出报告 | 最终 HTML review report |
| 运行根目录 | `.evocast/` |

## Evaluation Protocol 🧪

本仓库中的 forecasting evaluation workflow 遵循 [TFB](https://github.com/decisionintelligence/TFB) 实现的 benchmark protocol。EvoCast 在该 forecasting evaluation setup 之上完成 autonomous research、受边界约束的 repository-level implementation 和迭代状态更新。

## End-to-End Workflow 🚀

### 1. 配置 Provider 访问

为所选 provider 设置 API key。

```bash
# DeepSeek
export DEEPSEEK_API_KEY="your-api-key"

# MiniMax
export MINIMAX_API_KEY="your-api-key"

# OpenAI-compatible provider
export OPENAI_API_KEY="your-api-key"
```

在 Windows PowerShell 中：

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
$env:MINIMAX_API_KEY="your-api-key"
$env:OPENAI_API_KEY="your-api-key"
```

Provider 模板位于 `evocast/configs/providers/`。通过 `--api-config` 选择 provider 配置，例如 `providers/deepseek.yaml`、`providers/minimax.yaml` 或 `providers/openai.yaml`。

### 2. 启动交互式向导

```bash
python -m evocast
```

向导会收集任务配置、baseline、provider 和 research budget，然后启动工作流。

常用控制选项：

```text
--configure-only           只写入任务配置，不启动 research
--dry-run                  只校验和编译配置，不写运行时状态
--resume                   从已有 canonical state 继续
--max-rounds N             设置最大 research rounds
--dataset-diagnosis-mode   选择 required、reuse 或 skip
--baseline-strategy        选择自动 baseline 搜索或手动指定
--api-config               选择 provider 配置
```

### 3. 建立 Baseline

EvoCast 会编译任务契约，解析 baseline 策略，在固定 forecasting protocol 下执行 baseline，并写入初始任务状态。

输出路径：

```text
.evocast/task_knowledge/<task_id>/
```

### 4. 运行受控 Research Rounds

每个 research round 对应一个研究想法。系统会在隔离 workspace 中落实受边界约束的源码修改，验证候选，执行 canonical evaluation，并更新持久化任务状态。

核心运行链如下：

```text
evocast.scripts.wizard
  -> ResearchLoopService
  -> ResearchContractService
  -> run_agent_v3
  -> VariantForgeBackend
  -> ResearchBuildOrchestrator
  -> BuildAttemptEvaluator and TFBExperimentMetricRunner
  -> EvaluationDecisionKernel
```

### 5. 查看输出结果

运行时状态写入 `.evocast/`：

```text
.evocast/
  task_knowledge/<task_id>/
    domain_state.json
    runtime_events.jsonl
    rounds/<research_id>/
  dataset_knowledge/
  baseline_knowledge/
  runs/<task_id>/
  sandboxes/<task_id>/
  result/
  agent_reports/
```

最终报告会汇总 task configuration、baseline results、candidate attempts、validation outcomes、metrics、resource usage、round history 和 promotion decisions。

## Verification ✅

在项目根目录运行：

```bash
python -m compileall -q evocast
python -m pytest -q
git diff --check
```

## Acknowledgments 🙏

我们感谢 [TFB](https://github.com/decisionintelligence/TFB) 的作者开源统一的时间序列预测 benchmark 与 evaluation framework。EvoCast 将该 benchmark infrastructure 用作仓库中的 forecasting evaluation backbone。
