# MA-0 基线冻结与契约测试记录

> 对应计划：`unified_multi_agent_ui_plan.md`  
> 阶段：MA-0  
> 日期：2026-08-25  
> 状态：已验收

## 1. 冻结的现有入口

以下入口在 MA-0 期间保持不变：

```powershell
# 数据库子 Agent 独立页面
uv run materials-screen master db-ui

# 文献子 Agent 独立页面
uv run materials-screen literature ui --port 8504

# 现有 Master 多智能体 CLI
uv run materials-screen master ask "用户问题"
```

MA-1 前不新增统一页面，不改变上述入口的业务行为。

## 2. 冻结的现有多智能体组件

- `SubAgentSpec`：领域子 Agent 的静态能力与 runner 工厂；
- `SubAgentRegistry`：白名单注册、重复名称拒绝和委派函数解析；
- `MasterAgentRunner`：Master 会话与图执行入口；
- `materials_database`：数据库查询领域子 Agent；
- `literature`：文献与知识领域子 Agent。

## 3. MA-0 新增应用契约

实现文件：

```text
src/materials_screening/master/application_contracts.py
```

新增契约：

| 契约 | 职责 |
|---|---|
| `MultiAgentArtifact` | 统一引用领域 Agent 拥有的文件或结果对象 |
| `UnifiedResultReference` | 会话中保存的小型稳定结果指针 |
| `UnifiedConversationContext` | UI 与 Master 共享的可序列化上下文 |
| `UnifiedResultEnvelope` | 编排层到展示层的严格结果边界 |
| `SubAgentUiSpec` | 子 Agent 的声明式 UI 元数据 |
| `SubAgentUiRegistry` | UI 元数据的静态、重复安全注册表 |

所有 Pydantic 契约均采用：

- `extra="forbid"`；
- `frozen=True`；
- 有界字符串和安全名称；
- 重复 Artifact/引用拒绝；
- 澄清状态与追问字段一致性校验。

## 4. 明确未实现的内容

MA-0 没有实现：

- 统一 Gradio 页面；
- Master 自动路由改造；
- Artifact 文件复制或持久化服务；
- Renderer Registry；
- 子 Agent 跨域调用；
- 数据库 Schema 迁移；
- 现有子 Agent 业务修改。

## 5. 测试范围

新增测试：

```text
tests/unit/master/test_application_contracts.py
```

覆盖：

- 四类核心契约拒绝额外字段；
- Artifact JSON 往返；
- 会话 Artifact 唯一性；
- `last_results` 所有者一致性；
- 澄清结果信封约束；
- Artifact/Evidence 引用唯一性；
- UI Spec 有界校验；
- UI Registry 排序、查询和重复注册拒绝；
- 既有 `SubAgentRegistry` 回归。

## 6. MA-0 验收命令

```powershell
uv run ruff check `
  src/materials_screening/master/application_contracts.py `
  src/materials_screening/master/__init__.py `
  tests/unit/master/test_application_contracts.py

uv run mypy `
  src/materials_screening/master/application_contracts.py `
  src/materials_screening/master/__init__.py

uv run pytest `
  tests/unit/master/test_application_contracts.py `
  tests/unit/master/test_sub_agent_registry.py `
  tests/unit/master/test_material_database_routing.py `
  tests/unit/test_literature_ui_gradio.py -q
```

## 7. 进入 MA-1 的门槛

只有满足以下条件才能进入 MA-1：

- 上述 Ruff、mypy 和 pytest 全部通过；
- 现有数据库与文献定向回归通过；
- 新契约从 `materials_screening.master` 稳定导出；
- 没有修改现有子 Agent 的业务实现；
- 用户确认继续进入 MA-1。

## 8. 验收结果

2026-08-25 完成定向验收：

- Ruff：通过；
- mypy：通过；
- pytest：46 项通过；
- 覆盖新应用契约、既有 SubAgentRegistry、数据库路由和 Literature UI；
- 未修改 MaterialsDatabaseAgent 或 LiteratureAgent 的领域业务实现。

结论：MA-0 达到计划书验收条件。下一阶段为 MA-1，需由用户确认后开始。
