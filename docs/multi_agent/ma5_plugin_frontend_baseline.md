# MA-5 插件化扩展与前端稳定化验收记录

> 对应计划：`unified_multi_agent_ui_plan.md`  
> 阶段：MA-5  
> 日期：2026-08-25  
> 状态：已验收

## 1. 插件化边界

新增：

- `SubAgentUiPlugin`：声明 UI spec 与应用适配器；
- `SubAgentUiPluginRegistry`：按 Agent 名称和显示模式注册、解析；
- 统一页面的模式列表、快捷提示和显式分发均从插件注册表生成；
- MaterialsDatabaseAgent 和 LiteratureAgent 已迁移为两个内置插件；
- 自动 Master 路由、Artifact 路由和跨 Agent 工作流保持独立。

新增第三个子 Agent 时，只需：

1. 创建 `SubAgentUiSpec`；
2. 提供符合统一 handler 签名的适配器；
3. 注册 `SubAgentUiPlugin`；
4. 注册对应 Renderer（若使用新结果类型）。

不需要修改既有数据库或文献子 Agent。

## 2. 前端稳定化

- 增加“停止”按钮，可取消当前 Gradio 发送/提交事件；
- 领域网络请求和文献子进程继续使用已有超时边界；
- 插件异常由统一分发边界捕获，不显示内部异常消息，会话仍可继续；
- 初始化失败返回安全诊断，不破坏页面；
- 增加专家审计模式；
- 专家审计显示模式、最近 Agent、Master Conversation ID、Artifact 数量和文档数量；
- 审计视图明确不提供内部文件路径；
- 保留窄屏自动换行和三栏桌面布局；
- 新建会话会同步清理审计内容，但不会删除持久化结果。

## 3. 第三个 Agent 验证

自动测试注册了最小 `test_agent`：

- 新模式自动进入注册表；
- 主 `_dispatch` 无需增加 `if test_agent` 分支；
- 请求成功进入测试 handler；
- 状态记录 `last_agent=test_agent`；
- 重复名称和重复显示名称失败关闭；
- 测试 Agent 抛出异常时，统一页面返回可重试提示且不泄露异常详情。

## 4. 验收结果

- Ruff：通过；
- mypy：通过；
- Master、插件、数据库 UI、文献 UI、Artifact 和跨 Agent 定向测试：109 项通过；
- 手动数据库插件真实离线联调：通过；
- 手动文献插件真实离线联调：通过；
- 自动 Master 路由回归：通过；
- 页面启动：HTTP 200；
- 页面包含专家审计模式和停止按钮；
- 插件模式列表由注册表生成。

## 5. 启动

```powershell
uv run materials-screen master ui --port 8501
```

至此，`unified_multi_agent_ui_plan.md` 中 MA-0 至 MA-5 的最小正式闭环均已完成。
