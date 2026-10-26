# 极小报告预览

`source.py` 只有一个带 `if` 分支的函数。使用项目自身的确定性生成方式运行，未调用大模型。

```powershell
python main.py --source examples/minimal_report_preview/source.py --out examples/minimal_report_preview/report --knowledge-store examples/minimal_report_preview/knowledge.json --no-ui --max-iterations 1 --retry-rounds 0 --time-budget-seconds 120
```

本次结果：6 条测试全部通过；LC、BC 均为 100%，AE 为 90%，MS 为 50%，Pass Rate 与 TIR 均为 100%，Time 为 75.84 秒。AvgExec 因一次工具执行缺少完整计数证据而显示“未执行”，不代表执行次数为零。

- `report/state_flow_graph.html`：单任务图形报告。
- `report/source_coverage.html`：源码覆盖和测试证据页面。
- `report/stateflow_summary.json`：公开单任务结果。
- `report/state_flow_graph.json`：内部状态图，保留调度过程指标，不作为公开结果展示。
- `screenshots/`：实验总览、任务页、状态中心及两份 HTML 报告的实际页面截图。
- `screenshots/13_valifix_page_desktop.png`、`13_valifix_page_mobile.png`：独立的 ValiFixTest 论文方法展示台页面。图中的四项目百分比是论文历史数据，与本示例的运行指标分开。
