# DA-7 跨 Agent 与最终验收报告

> 对应计划：`data_analysis_agent_technical_plan.md`  
> 阶段：DA-7  
> 日期：2026-08-31  
> 状态：已验收

## 1. 最终交付结论

DataAnalysisAgent 已按 DA-0～DA-7 顺序完成契约、安全数据集、统计、不可变清洗、
图表与报告、八个白名单工具、Master/UI 接入和三条结构化跨 Agent 流程。

最终用户能力包括：

- 在统一页面上传 CSV/JSON，并在自动或显式数据分析模式中连续提问；
- 执行数据质量、描述统计、相关性、六类统计检验、不可变清洗、六类图表和报告；
- 通过稳定 Dataset/Analysis/Artifact/Evidence ID 追溯结果；
- 把 Materials Project 查询快照转成分析数据集；
- 把人工审核过的文献实验矩阵转成分析数据集；
- 把分析数据中的合法 `mp-*` ID 回查 Materials Project；
- 由 Master Renderer 按来源分区综合，不混写数据库值、文献测量值和用户数据。

## 2. 三条跨 Agent 验收链

### 2.1 MaterialsDatabaseAgent → DataAnalysisAgent

```text
query_id
  → QueryResultStore.load_records
  → 去除 structure/provenance 大对象并展平安全字段
  → immutable dataset_id
  → DataStatisticsService
```

只读取已保存快照，不重新查询数据库，不传内部路径。`query_id` 保留为来源证据。

### 2.2 LiteratureAgent → DataAnalysisAgent

```text
ExperimentMatrixDataTable
  → group/measurement 均为 approved
  → numeric measurement + chunk/page evidence
  → immutable dataset_id
  → DataStatisticsService
```

未审核记录、未知 group、无数值矩阵均关闭失败。同指标多单位不会自动换算，而是保留
原值并产生警告。

### 2.3 DataAnalysisAgent → MaterialsDatabaseAgent

```text
dataset_id + material_id column + analysis_ids
  → 严格验证、去重、最多 100 个 mp-* ID
  → MaterialDatabaseService.details
  → source-partitioned Master synthesis
```

非法 ID 不会传给数据库；数据库回查结果与数据分析结论使用不同来源区块。

## 3. 失败与部分成功

- 跨 Agent 不传自由文本大表、模型生成路径或未经验证的结构；
- 任一适配失败返回明确错误，不生成伪 Dataset 或伪查询；
- `SourcePartitionedSynthesis` 可返回 `partial`，保留已完成分区并列出失败警告；
- 跨来源 Renderer 固定展示来源类型和稳定 source ID；
- 普通回答和专家审计均不显示内部路径。

## 4. 最终验收门禁

```powershell
.venv\Scripts\python.exe -m ruff check <DA-0～DA-7 相关源文件与测试>

.venv\Scripts\python.exe -m mypy `
  src/materials_screening/data_analysis `
  src/materials_screening/sub_agents/data_analysis `
  src/materials_screening/master/artifact_registry.py `
  src/materials_screening/master/data_analysis_handoffs.py `
  src/materials_screening/master/result_renderers.py `
  src/materials_screening/master/ui_controller.py `
  src/materials_screening/master/mock_model.py `
  src/materials_screening/master/master_nodes.py `
  src/materials_screening/unified_ui_gradio.py

.venv\Scripts\python.exe -m pytest -q `
  tests/unit/data_analysis `
  tests/unit/sub_agents/test_data_analysis_agent.py `
  tests/unit/agent/test_tool_registry.py `
  tests/unit/agent/test_agent_security.py `
  tests/unit/master `
  tests/unit/test_unified_ui_gradio.py `
  tests/unit/agent/test_materials_database_ui.py `
  tests/unit/test_literature_ui_gradio.py `
  tests/unit/test_cli_version.py `
  --basetemp tmp/pytest-da7-gate
```

结果：

- Ruff：通过；
- mypy：22 个 DA-0～DA-7 相关源文件无类型错误；
- pytest：237 项通过；
- 三条允许的跨 Agent 流程均有端到端测试；
- 数据上传、自动/显式路由、连续会话、停止、错误恢复、路径隐藏、PDF、数据库和
  文献页面定向回归通过；
- `.pytest_cache` 权限产生 1 条缓存警告，不影响隔离测试结果。

## 5. 文档与限制

- 技术路线：`data_analysis_agent_technical_plan.md`；
- 用户指南：`data_analysis_agent_usage.md`；
- Agent README：`src/materials_screening/sub_agents/data_analysis/README.md`；
- 阶段基线：`da0_contract_baseline.md` 至本报告。

Excel、Parquet、数据库直连、任意代码、机器学习、自动方法选择、单位自动换算和因果
推断不在本 MVP 范围内。新增能力必须另立计划，不通过放宽当前安全边界实现。

## 6. 最终结论

DA-0～DA-7 全部达到计划书门槛，DataAnalysisAgent MVP 可以作为统一多 Agent 应用
中的正式子 Agent 使用。
