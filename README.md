# TSGColAgent

**状态引导的多智能体单元测试生成工具（Python / Java，本地开源模型）**

给定一个源码文件或项目目录，TSGColAgent 会分析程序结构、生成单元测试、执行并度量覆盖率与变异得分，
并根据“当前测试还缺什么”决定下一轮的目标——而不是一次性让模型猜一段测试。

> 本仓库同时提供论文方法的完整实现、可复现实验脚本，以及一个开箱即用的 Web 工作台 `tsgtest`。

---

## 为什么

自动化单元测试生成的难点不在“写出一段测试”，而在于三件事之间的持续衔接：

1. **测试目标**：当前代码哪些位置、分支、行为还没被检查；
2. **执行结果**：生成的测试是否真的跑通，覆盖到了什么，断言是否有效；
3. **历史经验**：上一次为什么失败，哪些做法在相似任务上有效。

多数工具把这三者分别放在提示词、日志和重试里，导致“覆盖不足 → 补测 → 断言薄弱 → 再补”
的过程难以连续。TSGColAgent 把程序结构、测试要求、执行结果和任务历史统一表示为**共享测试状态**，
由状态变化驱动后续目标。

## 方法要点

- **共享测试状态**：把函数、分支、断言、变异体映射到具体程序位置，记录每条测试要求的 Satisfied / Unsatisfied / Unknown / N/A；
- **测试状态引导（TSG）**：优先处理执行阻塞，再按前置条件与近期变化排序目标；候选必须保住已满足的要求；
- **三个智能体**：测试状态智能体（分析 + 规划）、测试生成智能体（生成 / 修复 / 优化）、测试知识智能体（跨任务经验检索）；
- **知识三态**：成功 / 失败 / 待验证经验，按后续任务的实际结果更新可靠度；
- **本地开源模型**：通过 Ollama 调用，默认 `deepseek-r1:14b`，可换成任意本地模型。

## 快速开始

### 1. 环境

- Python 3.11+
- [Ollama](https://ollama.com/)（本地模型服务）
- Java 任务另需 JDK 11+ 与 Maven 3.8+

```bash
pip install -r requirements.txt
ollama serve
ollama pull deepseek-r1:14b
```

### 1'. Docker 一键启动（评审推荐）

```bash
# 默认拉取 qwen2.5-coder:7b（约 4.7 GB）；显存紧张可换 1.5b
DEMO_MODEL=qwen2.5-coder:7b docker compose up -d
# 需要跑 Java 任务时带上 JDK + Maven：
WITH_JAVA=1 docker compose up -d --build
```

打开 `http://127.0.0.1:8770`。首次启动时 `model-pull` 会下载模型，`docker compose logs -f model-pull` 可看进度。

### 2. 启动工作台

```bash
python -m tsgtest serve --port 8770 --open
```

浏览器打开 `http://127.0.0.1:8770`：选择源文件/目录 → 选模型 → 开始生成，
右侧实时显示执行日志、覆盖率、变异得分与生成的测试代码。

> 没有模型也能看：点界面上的「查看示例运行（无需模型）」，会加载仓库内置的脱敏运行结果。

也可以直接用命令行：

```bash
python -m tsgtest gen demo/source.py -o runs/demo
python -m tsgtest gen path/to/src --json
```

环境自检：

```bash
python -m tsgtest check
```

### 3. 离线查看示例

仓库内 `demo/` 是一份完整的单任务运行结果（状态图、覆盖率报告、指标明细），
不启动模型也能打开工作台查看，用于了解输出形态。

## 仓库结构

```
agents/            三智能体与规划、评估、修复
graph/             共享测试状态、状态图、经验库
orchestration/     LangGraph 状态机
executor/          pytest / JUnit / JaCoCo / PIT 执行环境
metrics/           LC / BC / AE / MS / TIR 等指标（paper-v1 口径）
monitoring/        实验面板服务与页面数据
visualization/     状态图与知识库可视化
tsgtest/           开箱即用的 CLI + Web 工作台
frontend/          实验面板前端（React，含构建产物）
tests/             测试
demo/              一份完整运行示例（已脱敏）
```

## 复现实验

论文使用的数据集（TestEval、HumanEval、HumanEval-X、Defects4J）**不随仓库分发**，
请按各自许可自行获取，并按 `run_all_datasets.py` 的目录约定放置：

```
datasets/TestEval/  datasets/HumanEval/  datasets/HumanEvalJava/
datasets/Defects4J/{cli,csv,lang3,gson}/
```

```bash
python run_all_datasets.py --datasets-root datasets --out-root runs --llm --llm-model deepseek-r1:14b
```

指标导出与审计：

```bash
python scripts/export_paper_metrics.py --help
python scripts/audit_experiment_metrics.py --help
```

## 开源与合规

- 许可证：[MIT](LICENSE)
- 第三方资源、版本、许可证与自主实现边界：[THIRD_PARTY.md](THIRD_PARTY.md)
- 贡献方式：[CONTRIBUTING.md](CONTRIBUTING.md)
- 仓库不含模型权重、数据集、访问凭据与个人信息；`demo/` 中的历史路径已替换为占位符。

## 引用

如果本工具对你的研究或工程有帮助，可引用对应论文：

```bibtex
@article{tsgcolagent2026,
  title   = {基于测试状态引导机制的多智能体协作测试代码智能生成方法},
  author  = {TSGColAgent Authors},
  year    = {2026}
}
```

## 致谢

实现建立在 langgraph、tree-sitter、pytest、coverage、PIT、JaCoCo 等开源项目之上，
具体版本与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
