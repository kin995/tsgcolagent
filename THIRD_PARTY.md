# 第三方开源资源清单

本文件对应竞赛材料《开源及第三方资源使用清单》。**提交前请逐项复核官方 LICENSE**：
下表“状态”一列中，`已确认` 表示常见且稳定的许可证信息，`待核实` 表示需要以对应项目仓库的
LICENSE 文件为准（版本、模型变体、数据集条款可能不同）。

## 1. 运行时依赖（随代码分发或安装）

| 名称 | 版本 | 来源 | 许可证 | 使用方式 | 关键义务 | 状态 |
|---|---|---|---|---|---|---|
| langgraph | >=1.0,<2.0 | https://github.com/langchain-ai/langgraph | MIT | 多智能体状态机编排 | 保留版权与许可声明 | 已确认 |
| langgraph-checkpoint-sqlite | >=3.0,<4.0 | 同上 | MIT | 运行状态持久化 | 同上 | 已确认 |
| tree-sitter | >=0.25,<0.26 | https://github.com/tree-sitter/tree-sitter | MIT | Java 语法解析 | 同上 | 已确认 |
| tree-sitter-java | >=0.23,<0.24 | https://github.com/tree-sitter/tree-sitter-java | MIT | Java 语法定义 | 同上 | 已确认 |
| ollama (Python client) | >=0.4,<1.0 | https://github.com/ollama/ollama-python | MIT | 本地模型调用 | 同上 | 已确认 |
| pytest | >=8.0 | https://github.com/pytest-dev/pytest | MIT | 测试执行 | 同上 | 已确认 |
| coverage | >=7.0 | https://github.com/nedbat/coveragepy | Apache-2.0 | 行/分支覆盖采集 | 保留 NOTICE、许可证副本 | 已确认 |
| sortedcontainers | >=2.4,<3.0 | https://github.com/grantjenks/python-sortedcontainers | Apache-2.0 | 有序集合 | 同上 | 已确认 |

## 2. 前端依赖

| 名称 | 版本 | 许可证 | 使用方式 | 状态 |
|---|---|---|---|---|
| react / react-dom | ^18.3 | MIT | 界面框架 | 已确认 |
| vite | ^5.4 | MIT | 构建工具 | 已确认 |
| @vitejs/plugin-react | ^4.3 | MIT | Vite 插件 | 已确认 |
| typescript | ^5.7 | Apache-2.0 | 类型检查 | 已确认 |
| lucide-react | ^0.468 | ISC | 图标 | 已确认 |
| vitest / jsdom | ^2.1 / ^25 | MIT | 前端测试 | 已确认 |
| @testing-library/react、@testing-library/jest-dom | ^16 / ^6 | MIT | 前端测试 | 已确认 |

> 前端构建产物 `frontend/dist` 由上述依赖打包生成，分发时一并保留本清单与许可证。

## 3. 仅调用、不随仓库分发的工具链

| 名称 | 版本 | 许可证 | 使用方式 | 状态 |
|---|---|---|---|---|
| JUnit 5 | 5.12.0 | EPL-2.0 | 执行 Java 测试 | 已确认 |
| JaCoCo | 0.8.8 | EPL-2.0 | Java 覆盖率 | 已确认 |
| PIT (pitest) | 1.15.0 | Apache-2.0 | Java 变异测试 | 待核实 |
| Maven | 3.8.6 | Apache-2.0 | Java 构建 | 已确认 |
| JDK | 11 / 21 | GPLv2+CE | 运行环境 | 已确认 |
| Python | 3.11+ | PSF-2.0 | 运行环境 | 已确认 |

这些工具由用户环境自行安装，仓库不包含其二进制或源码，仅在文档中说明版本。

## 4. 模型（不随仓库分发）

| 名称 | 许可证 | 说明 | 状态 |
|---|---|---|---|
| DeepSeek-R1 系列 | MIT（以官方模型卡为准） | 通过 Ollama 拉取 | 待核实 |
| Qwen 系列 | 多为 Apache-2.0（部分变体不同） | 通过 Ollama 拉取 | 待核实 |
| gpt-oss-20b | Apache-2.0 | 通过 Ollama 拉取 | 待核实 |

仓库不包含模型权重；用户需自行同意模型许可后下载。

## 5. 数据集（不随仓库分发）

| 名称 | 用途 | 状态 |
|---|---|---|
| TestEval | Python 测试生成评测 | 待核实 |
| HumanEval | 函数级 Python 任务 | 已确认（MIT） |
| HumanEval-X | 函数级 Java 任务 | 待核实 |
| Defects4J | 项目级 Java 任务 | 待核实 |

**仓库不重新分发这些数据集。** 复现脚本只提供获取入口与目录约定，使用者需自行遵守各自许可。

## 6. 自主实现边界

以下内容为本项目自主实现，不在任何第三方许可证覆盖范围内：

- 共享测试状态（Test State）的数据结构与状态演化逻辑；
- 测试状态引导（TSG）的目标排序、已满足要求保护、候选比较规则；
- 测试状态 / 测试生成 / 测试知识三个智能体的职责划分与协作流程；
- 知识条目三态（成功 / 失败 / 待验证）验证与更新机制；
- 论文指标口径 `paper-v1` 的计算实现；
- 本仓库的 Web 工作台与命令行工具 `tsgtest`。

第三方依赖仅作为编排、解析、执行与度量基础设施，未修改其源码。

## 7. AI 工具使用说明

本项目在开发过程中使用了生成式 AI 辅助（代码补全、文档起草、代码审查）。所有 AI 生成
内容均经团队成员阅读、修改与测试验证后才纳入仓库；AI 生成内容不主张独立知识产权。
