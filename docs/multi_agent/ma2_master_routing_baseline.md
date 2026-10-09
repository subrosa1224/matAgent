# MA-2 MasterAgent 自动路由验收记录

> 对应计划：`unified_multi_agent_ui_plan.md`  
> 阶段：MA-2  
> 日期：2026-08-25  
> 状态：已验收

## 1. 本阶段实现

- 统一页面的“自动判断”模式接入真实 `MasterAgentRunner.ask_stream()`；
- 自动模式使用统一 Master Conversation ID，并在同一会话中持续复用；
- 路由、子 Agent 执行、校验和完成事件流式显示在执行轨迹中；
- 数据库请求委派 `materials_database`；
- 文献请求委派 `literature`；
- 不明确请求由 Master 安全澄清；
- 手动数据库与手动文献模式继续保留；
- 新增与前端框架无关的 `master/ui_controller.py`；
- 修复跨轮结果识别，避免上一轮子 Agent 被误当成本轮路由结果；
- Master 元数据检索不再无故加载 PDF 向量模型。

## 2. 本阶段边界

MA-2 仍未实现：

- PDF Artifact 自动路由；
- “第1篇”“这些论文”等附件指代；
- 一个请求同时调用多个子 Agent；
- 跨 Agent 材料线索传递；
- 通用 Artifact 持久化。

自动模式收到 PDF 时会安全提示切换到手动文献模式，不会把 PDF 传给数据库。

## 3. 验收结果

- Ruff：通过；
- mypy：通过；
- Master、路由与统一 UI 定向测试：63 项通过；
- 路由评测集：12/12 正确，准确率 100%；
- 真实离线 MasterRunner 数据库委派：通过；
- 真实离线 MasterRunner 文献委派：通过；
- 同一统一会话先数据库、后文献：路由正确且 Conversation ID 保持一致；
- 不明确请求：不调用子 Agent，返回安全澄清。

## 4. 启动命令

```powershell
uv run materials-screen master ui --port 8501
```

初始化后保持“自动判断”即可。需要诊断或处理 PDF 时，可手动选择对应子 Agent。
