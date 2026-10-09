# 统一多智能体与可视化技术计划书

> 文档状态：第一版已完成（MA-0 至 MA-5 均已验收）  
> 基线版本：v1.0  
> 制定日期：2026-08-25  
> 适用项目：`materials-screening-core` / `matAgent`  
> 当前纳入子 Agent：`materials_database`、`literature`  
> 实施原则：后续工作严格按本计划的阶段、边界和验收标准推进。

## 1. 背景与目标

项目当前已经具备 MaterialsDatabaseAgent、LiteratureAgent、`SubAgentSpec`、
`SubAgentRegistry` 和 `MasterAgentRunner` 等基础能力，但两个子 Agent 仍分别通过
独立网页使用，且 LiteratureAgent 网页仍包含部分前端意图路由和 CLI 适配逻辑。

本计划的目标不是把两个网页拼接，而是建立一个可持续扩展的多智能体应用：

1. 普通用户只面对一个统一对话入口；
2. MasterAgent 负责意图判断、任务拆分、子 Agent 调度和最终答案组织；
3. 专业子 Agent 保持独立的工具、数据、提示词、会话和测试边界；
4. 支持单 Agent 请求和跨 Agent 复合请求；
5. 后续增加新子 Agent 时以注册方式接入，不修改既有子 Agent；
6. 前端可修改或替换，而不影响 Agent 和 Service 层。

## 2. 范围与非目标

### 2.1 本计划范围

- 统一多智能体 Gradio 网页；
- MasterAgent 自动路由与显式路由模式；
- MaterialsDatabaseAgent 和 LiteratureAgent 接入；
- PDF 附件的安全接收和文献任务绑定；
- 统一会话、执行轨迹、结果信封和渲染注册表；
- 子 Agent 间结构化中间结果传递；
- 跨 Agent 任务的最小可用闭环；
- 单元测试、集成测试、验收测试和操作文档。

### 2.2 暂不纳入

- 删除或重写现有独立子 Agent 网页；
- 将两个子 Agent 的数据库合并为一个数据库；
- 让子 Agent 直接相互调用；
- 允许模型绕过工具层直接访问外部数据库；
- 在第一阶段实现任意数量 Agent 的自由自治协商；
- 公开互联网部署、用户账号、权限系统和多租户；
- MinerU、图表 OCR 等 LiteratureAgent 后续增强能力。

## 3. 强制架构原则

### 3.1 分层

```text
Unified UI
  └─ Unified Application Controller
       └─ MasterAgentRunner
            ├─ MaterialsDatabaseAgent
            │    └─ Materials Project / deterministic services
            └─ LiteratureAgent
                 └─ OpenAlex / Semantic Scholar / pgvector / PDF services
```

- UI 层只负责输入、附件、流式状态和展示；
- Application 层负责会话和附件上下文；
- Master 层负责路由、编排和最终汇总；
- SubAgent 层负责领域推理与工具选择；
- Service 层负责确定性访问、校验和持久化。

### 3.2 不允许的实现

- 在统一前端用关键词 `if/elif` 代替 MasterAgent 路由；
- 在前端直接拼接 CLI 原始输出作为普通用户回答；
- 在 MaterialsDatabaseAgent 中导入 LiteratureAgent 业务代码，反之亦然；
- 用自由文本作为跨 Agent 的唯一数据协议；
- 把本地路径、API Key、内部异常栈或风险计数展示在普通用户正文；
- 为了跨 Agent 综合而降低既有证据校验规则；
- 修改一个子 Agent 时要求其他子 Agent 同步修改。

## 4. 目标组件

### 4.1 Unified UI

计划新增：

```text
src/materials_screening/unified_ui_gradio.py
```

页面采用现有数据库子 Agent 的三栏对话布局：

- 左侧：运行配置、Agent 模式、附件、新建会话和系统状态；
- 中间：统一聊天、快捷示例、停止按钮；
- 右侧：Master 路由、子 Agent 调用、工具执行和证据状态；
- 普通模式隐藏内部标识；专家模式按需展开审计信息。

统一入口计划为：

```powershell
uv run materials-screen master ui
```

独立入口继续保留：

```powershell
uv run materials-screen master db-ui
uv run materials-screen literature ui
```

### 4.2 Unified Application Controller

控制器负责：

- 创建或恢复统一 Conversation ID；
- 安全复制和验证附件；
- 将附件登记为通用 Artifact；
- 调用 MasterAgent 流式接口；
- 把内部事件转换为用户执行轨迹；
- 调用对应 Renderer 输出最终内容。

控制器不得包含领域检索、论文总结或材料筛选规则。

### 4.3 MasterAgent

MasterAgent 负责：

- 判断请求属于数据库、文献或复合任务；
- 在信息不足时提出一次必要的澄清问题；
- 为复合请求生成有界执行计划；
- 串行调用子 Agent；
- 验证子 Agent 输出 Schema；
- 汇总结果并保持来源归属；
- 不自行编造材料记录、论文事实或证据。

### 4.4 SubAgentRegistry

每个子 Agent 必须通过注册表接入，至少声明：

```python
class SubAgentSpec:
    name: str
    description: str
    system_prompt: str
    tool_definitions: tuple
    runner_factory: Callable
```

后续扩展 UI 所需元数据应放入独立注册结构，不污染领域 Spec：

```python
class SubAgentUiSpec:
    name: str
    display_name: str
    accepted_artifact_types: tuple[str, ...]
    quick_prompts: tuple[str, ...]
    renderer_name: str
```

## 5. 核心数据契约

### 5.1 通用 Artifact

```json
{
  "artifact_id": "artifact-...",
  "owner_agent": "literature",
  "artifact_type": "pdf_document",
  "domain_id": "doc-...",
  "display_name": "paper.pdf",
  "status": "ready",
  "metadata": {}
}
```

UI 和 Master 只能通过 Artifact 关联领域对象，不假设所有 Agent 都使用 `document_id`
或 `query_id`。

### 5.2 统一会话上下文

```json
{
  "conversation_id": "...",
  "active_agent": null,
  "selected_mode": "auto",
  "artifacts": [],
  "last_results": {},
  "pending_clarification": null
}
```

`last_results` 只保存稳定结果引用，不把大段工具输出重复塞入前端状态。

### 5.3 子 Agent 结果信封

```json
{
  "agent_name": "literature",
  "status": "completed",
  "result_type": "literature_report",
  "result": {},
  "artifact_refs": [],
  "evidence_refs": [],
  "warnings": [],
  "follow_up_question": null
}
```

所有 Renderer 根据 `result_type` 注册，不根据 Agent 名称写长串条件判断。

### 5.4 跨 Agent 材料线索

LiteratureAgent 向 Master 返回可供数据库查询的结构化线索：

```json
{
  "material_clues": [
    {
      "name": "58S bioactive glass",
      "formula_or_composition": "58 SiO2 / 33 CaO / 9 P2O5 mol%",
      "elements": ["Si", "Ca", "P", "O"],
      "constraints": {},
      "source_document_id": "doc-...",
      "source_page": 3,
      "evidence_id": "E-..."
    }
  ]
}
```

Master 将其转换成 MaterialsDatabaseAgent 的受约束输入，不能直接传递自由文本结论。

## 6. 路由规则

### 6.1 单 Agent 路由

- 材料筛选、详情、比较、统计、离群检测和导出 → `materials_database`；
- 文献发现、PDF 阅读、论文分析和综合报告 → `literature`；
- 用户显式选择 Agent 模式时，Master 仍需检查能力边界，但优先采用用户选择。

### 6.2 跨 Agent 路由

第一版只允许有界的串行计划：

```text
LiteratureAgent → Master validation → MaterialsDatabaseAgent → Master synthesis
```

或：

```text
MaterialsDatabaseAgent → Master validation → LiteratureAgent → Master synthesis
```

每轮最多调用两个领域子 Agent；不得无限循环协商。

### 6.3 澄清

仅在缺失信息会显著改变任务时澄清，例如：

- “查一下这个材料”，但上下文中没有材料；
- 用户要求综合论文，但未上传或选择论文；
- 用户要求跨数据库映射，但未说明是按元素、结构还是性能匹配。

## 7. 附件处理

- PDF 只允许进入授权目录；
- 校验扩展名、文件头、大小、解析页数和规范化路径；
- 原始用户文件名仅用于显示；
- Master 获取 Artifact 引用，不直接获取任意本地路径；
- 上传本身不自动批准实验数据；
- 无主题上传允许逐篇预览或分析；
- 只有主题筛选和主题综合才需要主题。

后续增加 CSV、CIF、JSON 等附件时，通过 Artifact Handler 注册，不修改聊天主流程。

## 8. 前端渲染与可扩展性

建立 Renderer Registry：

```python
RESULT_RENDERERS = {
    "materials_table": render_materials_table,
    "material_comparison": render_material_comparison,
    "literature_candidates": render_literature_candidates,
    "paper_dossier": render_paper_dossier,
    "literature_report": render_literature_report,
    "cross_agent_report": render_cross_agent_report,
}
```

普通用户输出要求：

- 只显示问题答案、必要页码和数据来源；
- 不显示 CLI 日志、内部路径、堆栈、批次 ID 和风险条数；
- 论文直接结论与跨论文/跨 Agent 综合判断必须明确区分；
- 引用具体数值时保留论文页码或数据库证据；
- 技术审计信息进入右侧轨迹或专家模式。

新增子 Agent 时，原则上只新增：

1. `SubAgentSpec`；
2. 可选 `SubAgentUiSpec`；
3. 新的 Result Schema；
4. 对应 Renderer；
5. 路由样例和测试。

## 9. 安全与证据

- 外部论文、摘要和 PDF 均视为不可信数据；
- 子 Agent 只能调用白名单工具；
- Master 不得把一个 Agent 的 evidence_id 冒充为另一个 Agent 的证据；
- 跨 Agent 最终结论必须保留双侧 provenance；
- 高风险或未通过校验的文献内容不得进入普通最终结论；
- API Key 只从环境读取，不写入状态、日志和页面；
- 用户附件不自动上传到第三方；
- 写操作、导出和未来外部发布遵循显式授权边界。

## 10. 分阶段实施计划

### MA-0：基线冻结与契约测试

目标：确认现有能力不回退，并冻结统一接口。

任务：

- 记录数据库和文献独立 UI 的可用命令；
- 记录现有 Master 注册流程；
- 定义 Artifact、Conversation、ResultEnvelope 和 UiSpec 模型；
- 增加 Schema 单元测试；
- 不修改业务行为。

验收：

- 现有两个独立 UI 均可启动；
- 现有相关测试通过；
- 四个核心 Schema 可序列化、校验并拒绝额外字段。

### MA-1：统一 UI 外壳

目标：建立一个聊天入口，但暂不启用自动路由。

任务：

- 新增统一 Gradio 页面；
- 增加 `auto/database/literature` 模式选择；
- 实现统一附件区、聊天区和轨迹区；
- 接入 Renderer Registry；
- 显式模式分别调用现有子 Agent；
- 保留独立 UI。

验收：

- 同一页面可完成数据库查询和论文分析；
- 用户问题立即显示；
- 普通回答不出现 CLI 原始日志；
- 新建会话会清理前端上下文但不删除持久化结果。

### MA-2：MasterAgent 自动路由

目标：由真实 MasterAgent 替代前端路由。

任务：

- 接入现有 `MasterAgentRunner`；
- 使用统一 Conversation ID；
- 流式展示路由和子 Agent 事件；
- 自动模式覆盖数据库、文献和澄清请求；
- 删除统一前端中的领域关键词路由。

验收：

- 路由评测集中数据库请求和文献请求准确率不低于 90%；
- 同一会话的后续指代可恢复正确上下文；
- 路由失败时安全澄清，不调用错误写工具；
- 手动模式仍可作为诊断回退。

### MA-3：Artifact 与附件上下文

目标：PDF 等附件成为统一会话的一等对象。

任务：

- 实现 Artifact Registry；
- PDF 上传生成 Artifact；
- LiteratureAgent 通过 Artifact 解析文件；
- 支持“第1篇”“这些论文”等指代；
- 附件状态进入右侧轨迹。

验收：

- 无主题 PDF 可直接预览和深度分析；
- PDF 不会错误传给 DatabaseAgent；
- 非授权路径、伪 PDF 和超限文件被拒绝；
- 页面不暴露真实内部路径。

### MA-4：跨 Agent 最小闭环

目标：完成第一类结构化复合任务。

首个用例：

> 从所选论文提取材料组成，再到 Materials Project 查询包含相应元素的稳定候选，
> 最后同时展示论文依据和数据库依据。

任务：

- 定义 MaterialClue Schema；
- LiteratureAgent 输出 MaterialClue；
- Master 校验并转换数据库查询输入；
- DatabaseAgent 返回候选快照；
- CrossAgentRenderer 生成双来源结果。

验收：

- 不通过自由文本传递材料约束；
- 每个候选同时标注材料数据库来源和论文来源；
- 文献组成不能被误当作 Materials Project 中的确定化合物；
- 任一子 Agent 失败时返回部分结果和明确边界。

### MA-5：插件化扩展与前端稳定化

目标：验证第三个子 Agent 可无侵入接入。

任务：

- 完成 SubAgentUiSpec 和 Renderer 自动注册；
- 用最小测试 Agent 验证注册流程；
- 增加专家模式和审计面板；
- 完善响应式布局、停止、超时和错误恢复；
- 更新 README 和运维文档。

验收：

- 新增测试 Agent 不修改既有子 Agent 和聊天主流程；
- 更换 Renderer 不影响 Agent 测试；
- 普通模式与专家模式信息边界清晰；
- 统一 UI 回归测试通过。

## 11. 测试策略

### 11.1 单元测试

- Schema 校验；
- Registry 注册和重复名称拒绝；
- Renderer 路由；
- Artifact 安全校验；
- 会话状态更新；
- 普通模式敏感内部信息过滤。

### 11.2 集成测试

- Master → DatabaseAgent；
- Master → LiteratureAgent；
- PDF Artifact → LiteratureAgent；
- LiteratureAgent → MaterialClue → DatabaseAgent；
- 子 Agent 部分失败和超时。

### 11.3 UI 验收场景

1. “筛选带隙大于 2 eV 的稳定氧化物”；
2. “检索 3D 打印生物活性玻璃支架的近年论文”；
3. 上传三篇 PDF 后说“分别预览这些论文”；
4. “深度分析第 1 篇”；
5. “综合第 1 和第 3 篇”；
6. “根据论文材料组成查询数据库候选”；
7. 同一会话追问“比较刚才前三个候选”；
8. 新建会话后确认旧指代不再生效。

## 12. 可观测性和错误处理

右侧轨迹使用稳定事件：

```text
received → routing → delegated → tool_running → validating → completed/partial/failed
```

普通用户看到中文动作描述；专家模式可查看：

- agent_name；
- tool_name；
- artifact_ref；
- evidence_ref；
- latency；
- stable error code。

错误必须转换为可行动提示。不得把 traceback 直接返回聊天框。

## 13. 兼容、迁移与回滚

- 不删除现有 CLI 和独立 UI；
- 统一 UI 使用新文件和新命令，减少对现有功能的侵入；
- 每阶段单独提交并通过验收后进入下一阶段；
- 自动路由失败时可切回手动 Agent 模式；
- 跨 Agent 功能失败不影响单 Agent 功能；
- 数据库迁移必须保持向后兼容，不在 UI 阶段修改领域表结构。

## 14. 变更控制

从本基线生效后：

1. 每次实施必须声明当前阶段编号；
2. 不跨阶段提前加入非必要能力；
3. 阶段完成必须附测试结果和未完成项；
4. 如需改变架构、Schema 或阶段范围，先修改本计划并记录原因；
5. 用户体验问题优先在 Renderer/UI 层解决，不破坏领域证据规则；
6. 不以“未来可能需要”为由提前实现大范围功能。

## 15. 完成定义

满足以下条件才视为统一多智能体第一版完成：

- 普通用户通过一个网页完成数据库和文献任务；
- MasterAgent 真实负责自动路由；
- PDF 通过 Artifact 管理；
- 至少一个跨 Agent 用例完成结构化闭环；
- 新增第三个测试 Agent 无需修改既有子 Agent；
- 普通回答不泄露内部日志和路径；
- 关键结论保留可追溯证据；
- 独立子 Agent 页面继续可用；
- 计划内测试和文档全部通过。

## 16. 实施结果

MA-0 至 MA-5 已依次完成并分别形成验收记录：

- `ma0_contract_baseline.md`
- `ma1_unified_ui_baseline.md`
- `ma2_master_routing_baseline.md`
- `ma3_artifact_baseline.md`
- `ma4_cross_agent_baseline.md`
- `ma5_plugin_frontend_baseline.md`

统一多智能体第一版已满足第 15 节的完成定义。后续新增领域 Agent 或调整前端时，
继续遵守本文件的契约边界、证据规则和变更控制要求。
