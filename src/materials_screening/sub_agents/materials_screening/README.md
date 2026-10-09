# Materials Screening 子 Agent

无机半导体材料筛选子 Agent — 多智能体系统中的 Sub-Agent 1。

## 能力

接收自然语言筛选需求，调用 5 个工具完成材料筛选：

| 工具 | 功能 |
|---|---|
| `run_screening_workflow` | 执行完整筛选管线 |
| `get_workflow_status` | 查询工作流状态 |
| `get_workflow_history` | 查询工作流执行历史 |
| `get_screening_result` | 读取筛选结果详情 |
| `compare_ranked_materials` | 对比多个候选材料 |

## 文件

| 文件 | 说明 |
|---|---|
| `tools.py` | 包装现有 `agent_tools/` 的 5 个工具到 `AgentToolRegistry` |
| `prompt.py` | 子 Agent 的 system prompt（材料筛选专家人设） |
| `spec_factory.py` | `create_spec()` — 产出 `SubAgentSpec`，供 Master 注册 |

## 在 Master 中注册

```python
from materials_screening.sub_agents.materials_screening.spec_factory import create_spec
spec = create_spec(workflow_runner=runner, intern_api_key="your-intern-token")
master.register_sub_agent(spec)
```
