# CacheScope MASC520 实验代码与数据

本项目研究一个具体问题：在调用回答模型之前，能否利用轻量神经网络识别可能损害答案准确率的 prompt 压缩，并在准确率与输入 token 用量之间取得更好的权衡。

系统对每个问题生成两份压缩候选，分别提取回答前特征，由风险模型打分；在通过验证集阈值的候选中选择最短的一份，两份均不通过时使用原文。MLP 是否优于简单规则、Logistic Regression 或 Random Forest，由保存的实验结果决定。

## 本次实验范围

| 项目 | 配置 |
| --- | --- |
| 开发数据 | 100 个独立合成任务，仅用于开发检查 |
| 正式数据 | 500 个合成任务，固定划分为 300 训练、100 验证、100 测试 |
| 额外结构留出集 | 50 个合成任务，包含表格、嵌套 JSON、双目录关联三种结构 |
| 补充长度压力测试 | 50 个既有测试来源的配对变体，添加无关附录后上下文为 1,000–1,013 tokens；不参与训练或阈值选择 |
| 候选压缩 | 同一个词汇重合句子选择器，目标保留 60% 或 30% 上下文 token |
| 回答模型 | 本地 Ollama `qwen2.5-coder:0.5b`，实际 digest 见 `results/answering_model.json` |
| 风险模型 | PyTorch MLP，另有 Logistic Regression 和 Random Forest 基线 |
| 重复与消融 | MLP 训练种子 520、521、522；移除数值／日期特征、移除问题重合特征 |

这些任务抽取项目负责人、发布日期或预算。文本、问题、参考答案均为本项目规则生成的合成数据；风险标签来自真实本地模型的原文／压缩回答与参考答案的评分。主数据上下文实测为187–247 tokens，属于短文本。另用约1,000-token的无关附录压力测试检查长度变化；这种受控填充不等于真实长文、真实用户请求或其他回答模型上的效果验证。

已测主要发现：主测试集上，MLP输入token减少32.60%，准确率由76%变为79%，准确率变化的分组95%区间跨越零；结构留出集上准确率由86%降至68%，说明当前风险模型存在明显迁移问题。完整数字和限制见 `RESULTS.md`，双击 `实验结果.html` 可查看表格与图。

## 从哪里查看结果

| 路径 | 内容 |
| --- | --- |
| `results/metrics.json` | 验证集选定阈值、测试集和结构留出集的任务级策略结果及不确定性 |
| `results/policy_results.csv` | 原文、固定压缩、规则门控和学习门控的汇总对照表 |
| `results/figures/` | 策略对比、准确率与 token 权衡、风险 precision–recall 图 |
| `results/task_policy_decisions.jsonl` | 各策略对每个任务实际选择了哪个候选 |
| `results/failures.jsonl` | 测试／结构留出集中错误或有害压缩的上下文、答案与决策 |
| `models/training_report.json` | 分类指标、样本数量、训练配置、软件版本及各模型结果 |
| `models/*_history.json` | MLP 训练与验证损失，包含三个种子和特征消融 |
| `data/raw_responses.jsonl` | 真实回答的追加缓存、用量、耗时和请求身份摘要 |
| `data/scored_candidates.jsonl` | 两档候选、标签、特征及各模型风险分数 |
| `results/answering_model.json` | 回答模型 digest、参数、prompt 说明和运行时信息 |
| `results/length_stress_metrics.json` | 固定模型与原验证阈值下的补充长度压力测试 |
| `docs/FAILURE_ANALYSIS.md` | 实际失败案例、消融和泛化不足分析 |

指标以实际生成的文件为准。正式评测每个任务只计一次，两档候选不被当作两个独立测试任务。MLP 不一定胜过其他方法；保留原文的策略也可能是验证集条件下合理的结果。

## 安装与运行

在本文件所在目录打开 PowerShell。测试使用 CPU；训练风险模型不要求 GPU。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

标签生成需要已安装并启动的 [Ollama](https://ollama.com/)。启动 Ollama 应用，或在另一个终端运行 `ollama serve`，然后下载模型：

```powershell
ollama pull qwen2.5-coder:0.5b
```

按顺序执行四个阶段：

```powershell
.\.venv\Scripts\python.exe run_experiment.py prepare --n-tasks 500
.\.venv\Scripts\python.exe run_experiment.py pilot
.\.venv\Scripts\python.exe run_experiment.py label
.\.venv\Scripts\python.exe run_experiment.py train
```

补充检查使用已有固定模型和阈值：

```powershell
.\.venv\Scripts\python.exe repeat_subset.py
.\.venv\Scripts\python.exe benchmark_decision.py
.\.venv\Scripts\python.exe length_stress.py prepare
.\.venv\Scripts\python.exe length_stress.py label
.\.venv\Scripts\python.exe length_stress.py evaluate
.\.venv\Scripts\python.exe verify_results.py
.\.venv\Scripts\python.exe build_summary.py
```

长度压力测试的50条变体与主测试来源配对，不是新增50个独立来源；不能将两批当作独立样本合并计算置信区间。重新生成的输入必须与 `results/length_protocol.json` 的哈希一致。

本机已配置环境可用 `powershell -ExecutionPolicy Bypass -File .\run_local.ps1 -Stage verify` 核对交付结果。预测示例：

```powershell
$questionText = Get-Content -Raw examples/question.txt
.\.venv\Scripts\python.exe predict.py --context-file examples/context.txt --question $questionText
```

该命令只做压缩决策，不调用回答模型；交付时的真实预测输出见 `examples/prediction.json`。

`prepare` 根据固定种子重建任务及划分；`pilot` 运行 100 个开发任务；`label` 运行正式数据和额外结构留出集；`train` 使用已经保存的候选标签训练并评测。标签阶段每个任务最多需要三个回答：原文、60% 候选、30% 候选，完全相同的请求会读取本地缓存。100 个开发任务、500 个正式任务和 50 个结构留出任务合计最多约 1,950 个不同请求，实际数目受候选是否相同等因素影响。

标签过程可中断后用相同命令续跑，已完成请求从 `data/raw_responses.jsonl` 读取。候选表会按任务重新构建。缓存与模型、选项和 prompt 有关；变更回答模型版本后应使用独立运行目录或新的缓存，以免把不同实验混为一谈。复现时比较 `results/answering_model.json` 中的 digest，而不能只比较可更新的 Ollama 模型标签。

已附带完整候选标签时，重新执行 `train` 不需要启动 Ollama。开发用 `--limit` 只适用于流程检查，不能将部分任务结果作为正式 500 任务实验；正式命令应省略此参数。

测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 文件结构

```text
assets/                  固定版本 Qwen tokenizer、来源哈希及原始许可
data/                    合成任务、固定划分、真实响应缓存、候选与风险分数
src/cachescope/           生成、压缩、评分、特征、训练、推理和策略评测代码
models/                  风险模型、预处理参数、分类报告和训练曲线数据
results/                 模型身份、策略指标、决策、失败案例及图表
docs/DATA_CARD.md        数据、标签、划分与适用范围
docs/METHODS.md          训练与评测方法及复现细节
tests/                   数据、评分、特征、训练和策略选择测试
run_experiment.py        命令行入口
requirements.txt         运行依赖
```

## 结果应如何解释

风险分类器只在“原文回答正确”的训练候选中学习“压缩后由对变错”，输出是未校准的条件风险分数。真实推理时不会先调用回答模型判断原文是否正确。最终策略在全部测试任务上评测，包括原文本来答错的任务，并分别统计有害压缩和压缩后改善。

完整 prompt token 用量包含系统指令、问题、上下文和 ChatML 包装。它与输入 API 金额、订阅额度、输出 token 或完整任务总消耗不是同一个指标。本实验没有测量云 API 金额或订阅额度。保存的回答耗时受请求顺序、模型加载和缓存影响；本地模型批量打分时间也不是完整在线延迟，不能据此宣称端到端加速。

正式数据的 50 个分组代表措辞模板家族；五类高层结构在训练、验证和测试中均出现。额外 50 个任务检查三种不同合成结构，但不构成人工独立样本或真实世界泛化证据。验证集上 2 个百分点的经验限制不是测试集安全保证。更多细节见两份方法文档。

## 来源与贡献说明

代码和合成数据生成规则使用 AI 辅助编写，实验响应由本地 Qwen 模型实际生成；参考答案来自已定义合成事实，未由风险模型预测。课程提交时应按课程政策披露实际 AI 工具、用途以及人工核验范围。两名成员的姓名和实际贡献应由成员据实补充，本仓库不预填个人贡献。

本项目生成的合成数据随代码提供用于课程复现，未引入第三方事实数据集。Qwen tokenizer 来自 [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)，版本、文件哈希见 `assets/tokenizer_source.json`，Apache 2.0 许可原文保存在 `assets/LICENSE`。回答模型权重由 Ollama 单独管理，不包含在项目目录内。

相关研究：[LLMLingua](https://aclanthology.org/2023.emnlp-main.825/)、[LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/)。本项目围绕固定压缩器的拒绝决策开展课程实验，未复现这些论文的模型或声称新的通用压缩算法。
