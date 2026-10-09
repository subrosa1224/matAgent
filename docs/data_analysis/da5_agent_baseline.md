# DA-5 DataAnalysisAgent 与白名单工具基线

> 对应计划：`data_analysis_agent_technical_plan.md`  
> 阶段：DA-5  
> 日期：2026-08-31  
> 状态：已验收

## 1. 本阶段交付

```text
src/materials_screening/sub_agents/data_analysis/
├─ __init__.py
├─ models.py
├─ tools.py
├─ prompt.py
├─ routing_model.py
├─ mock_model.py
├─ spec_factory.py
└─ README.md

tests/unit/sub_agents/test_data_analysis_agent.py
```

Agent 工具名称已加入静态 `ALLOWED_TOOL_NAMES`，相应安全白名单测试同步更新。

## 2. 八个白名单工具

| 工具 | 副作用 |
|---|---|
| `inspect_dataset` | 只读 |
| `assess_data_quality` | 只读 |
| `describe_dataset` | 只读 |
| `analyze_correlations` | 只读 |
| `run_statistical_test` | 只读 |
| `transform_dataset` | 创建派生结果 |
| `create_analysis_plot` | 创建 Artifact |
| `create_analysis_report` | 创建报告或显式数据导出 Artifact |

所有输入模型采用 `extra="forbid"` 和 `frozen=True`，嵌套 TransformOperation 同样
是严格结构化模型。工具结果不包含内部路径，只返回 Dataset、Inspection、Analysis、
Transform、Artifact 和 Evidence ID。

## 3. Agent 行为边界

- 禁止任意 Python、SQL、Shell、Notebook、eval 和动态代码；
- 陌生数据先检查，不把完整表格放入模型上下文；
- 不自动从描述统计升级为显著性检验；
- 不静默替换统计方法；
- 清洗、绘图、报告和导出必须由用户明确请求；
- 清洗永远产生新数据集；
- 数值结论必须来自工具并引用 Evidence ID；
- 数据集、字段、分组、配对或方法不足时只提出必要澄清。

## 4. 模型与运行工厂

- `DataAnalysisRoutingModel` 只对带完整 dataset ID 的明确检查/质量请求执行确定性
  首工具路由，其余请求交给领域模型；
- `DataAnalysisMockModel` 离线完成 dataset ID 澄清、检查/质量工具调用和证据化回答；
- `create_spec()` 创建独立 DatasetStore、工具注册表、会话 SQLite 和 checkpoint；
- 真实模式使用 Intern 模型，离线模式不访问网络；
- SubAgentSpec 名称为 `data_analysis`，委派函数为 `delegate_to_data_analysis`。

## 5. 验收命令与结果

```powershell
.venv\Scripts\python.exe -m ruff check `
  src/materials_screening/data_analysis `
  src/materials_screening/sub_agents/data_analysis `
  src/materials_screening/agent/tool_registry.py `
  tests/unit/data_analysis `
  tests/unit/sub_agents/test_data_analysis_agent.py `
  tests/unit/agent/test_agent_security.py

.venv\Scripts\python.exe -m mypy `
  src/materials_screening/data_analysis `
  src/materials_screening/sub_agents/data_analysis

.venv\Scripts\python.exe -m pytest `
  tests/unit/data_analysis `
  tests/unit/sub_agents/test_data_analysis_agent.py `
  tests/unit/agent/test_tool_registry.py `
  tests/unit/agent/test_agent_security.py `
  tests/unit/master/test_application_contracts.py `
  tests/unit/master/test_sub_agent_registry.py `
  -q --basetemp tmp/pytest-da5-run3
```

结果：

- Ruff：通过；
- mypy：15 个源文件无类型错误；
- pytest：141 项通过；
- `.pytest_cache` 权限产生 1 条缓存警告，不影响隔离测试结果。

## 6. DA-5 结论

DA-5 达到计划书的工具白名单、严格 Schema、Evidence、作用域、确定性路由、
离线 mock、runner factory、安全和前序回归门槛。下一阶段为 DA-6：以注册方式
接入 Master、CSV/JSON Artifact 和统一 UI。
