# DataAnalysisAgent

面向已登记 CSV/JSON 数据集的确定性数据分析子 Agent。

能力包括数据检查、质量画像、描述统计、相关性、显式统计检验、不可变清洗、
六类固定图表、Markdown/CSV/JSON 报告和安全导出。Agent 不执行任意代码，
不接收本地路径，数值结论必须来自白名单工具并关联 Evidence ID。

该目录只负责 Agent 契约、提示词、工具和 runner；解析、存储、统计、清洗和报告
实现位于 `materials_screening.data_analysis`。

统一应用通过 `delegate_to_data_analysis` 注册该 Agent。CSV/JSON 上传会先生成
`artifact-data-*` 与 `dataset-*`，显式“数据分析”模式和自动模式都只把稳定 ID
放入任务。数据库快照与人工审核文献矩阵必须通过
`DataAnalysisCrossAgentCoordinator` 的结构化适配，禁止复制自由文本或内部路径。

用户指南：`docs/data_analysis/data_analysis_agent_usage.md`。
