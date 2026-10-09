# DA-6 Master、Artifact 与统一 UI 接入基线

> 对应计划：`data_analysis_agent_technical_plan.md`  
> 阶段：DA-6  
> 日期：2026-08-31  
> 状态：已验收

## 1. 本阶段交付

- `data_analysis` 已加入 `_open_master_runner()` 的 `SubAgentRegistry`；
- Master 确定性路由和离线 mock 均优先识别 dataset ID、CSV、数据质量、
  缺失值、相关性、统计检验和实验数据等强信号；
- `ArtifactRegistry` 支持 PDF 与 CSV/JSON 两类附件，分别归属
  `literature` 和 `data_analysis`；
- CSV/JSON 在登记时经过格式、UTF-8、规模、表结构和嵌套数据校验，复制到
  私有目录后再登记为不可变 Dataset；
- 统一 UI 新增“数据分析”显式模式、四个快捷任务、CSV/JSON 上传、自动附件
  分流和 `analysis_answer` 渲染；
- `DatasetStore` 支持长生命周期 Reader 发现其他实例新登记的 Dataset，保证
  UI 上传后，已初始化的 DataAnalysisAgent 能按稳定 ID 读取；
- UI 只显示附件名和稳定 ID 上下文，不显示 Artifact 或 Dataset 内部路径。

## 2. 路由优先级与边界

```text
CSV/JSON 附件或明确 dataset/data-quality 信号 → DataAnalysisAgent
PDF 附件或论文引用                         → LiteratureAgent
Materials Project / mp-* / 材料筛选         → MaterialsDatabaseAgent
```

“分析”单字不作为数据分析路由依据。含“统计”的请求只有同时出现数据集、
CSV、实验数据等强信号时才进入 DataAnalysisAgent，因此数据库快照统计和论文
分析保持原有归属。混合上传 PDF 与 CSV/JSON 会被明确拒绝。

## 3. UI 行为验收

- 自动模式上传 CSV 后可完成登记、路由和数据检查；
- 显式数据分析模式通过 `SubAgentUiPluginRegistry` 注册；
- 连续请求复用 Master conversation ID 和当前 dataset ID；
- 非法 JSON 被拒绝后，同一会话可以改传合法 CSV 并恢复；
- Stop 按钮继续取消 send/submit 两个流式事件；
- 专家审计仅展示 Artifact 数量、Agent 和状态，不展示内部路径；
- PDF 自动预览、文献分页、材料数据库手动模式无行为回归。

## 4. 验收命令与结果

```powershell
.venv\Scripts\python.exe -m ruff check <DA-0～DA-6 相关源文件与测试>

.venv\Scripts\python.exe -m mypy `
  src/materials_screening/data_analysis `
  src/materials_screening/sub_agents/data_analysis `
  src/materials_screening/master/artifact_registry.py `
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
  --basetemp tmp/pytest-da6-gate2

.venv\Scripts\python.exe -m materials_screening.cli master sub-agents
```

结果：

- Ruff：通过；
- mypy：21 个源文件无类型错误；
- pytest：232 项通过；
- CLI 组装检查列出 `delegate_to_data_analysis` 及既有四个委派入口；
- `.pytest_cache` 权限产生 1 条缓存警告，不影响隔离测试结果。

## 5. DA-6 结论

DA-6 达到计划书关于 Master 注册、自动/mock 路由、Artifact 所有权、显式 UI
插件、上传安全、Renderer、连续会话、错误恢复、停止和路径隐藏的门槛。
下一阶段为 DA-7：只实现计划书允许的结构化跨 Agent 适配、端到端验收和最终文档。
