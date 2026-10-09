# MA-1 统一 UI 外壳验收记录

> 对应计划：`unified_multi_agent_ui_plan.md`  
> 阶段：MA-1  
> 日期：2026-08-25  
> 状态：已验收

## 1. 本阶段实现

- 新增统一 Gradio 页面 `unified_ui_gradio.py`；
- 新增 `materials-screen master ui` 入口；
- 提供“材料数据库”“文献与知识”和延后启用的“自动判断”模式；
- 复用现有 Database UI 与 Literature UI 控制器；
- 数据库和文献会话状态在统一页面中隔离保存；
- PDF 只允许传入文献模式；
- 新增 `ResultRendererRegistry`；
- 新增两个真实 `SubAgentUiSpec` 注册项；
- 保留两个独立子 Agent 页面和命令。

## 2. 架构边界

MA-1 没有实现：

- MasterAgent 自动路由；
- 前端关键词自动选择 Agent；
- 跨 Agent 任务；
- 通用 Artifact 持久化；
- PDF 传入数据库子 Agent；
- 删除或替换独立 UI。

选择“自动判断”时，页面只说明 MA-2 尚未启用，不执行请求。

## 3. 新增文件

```text
src/materials_screening/unified_ui_gradio.py
src/materials_screening/master/result_renderers.py
tests/unit/test_unified_ui_gradio.py
tests/unit/master/test_result_renderers.py
```

## 4. 验收要求

- Ruff、mypy 通过；
- 新增和既有定向测试通过；
- `master ui --help` 显示新命令；
- 页面对象可构建；
- 本地服务返回 HTTP 200；
- 显式数据库模式委派现有 Database UI 控制器；
- 显式文献模式委派现有 Literature UI 控制器；
- 自动模式不执行伪路由；
- 数据库模式拒绝 PDF；
- Renderer 对未知类型和重复注册失败关闭。

## 5. 启动命令

```powershell
uv sync --extra web-ui
uv run materials-screen master ui --port 8501
```

访问 `http://127.0.0.1:8501`。如端口占用，程序自动选择后续空闲端口。

## 6. 验收结果

- Ruff：通过；
- mypy（MA-0/MA-1 新增核心模块）：通过；
- 定向测试：29 项通过；
- `materials-screen master ui --help`：通过；
- 统一页面实际启动：通过；
- 本地 HTTP 检查：状态码 200，页面包含 Gradio 入口与 `Materials Multi-Agent` 标题；
- 验收端口 8507 已在检查结束后关闭。
