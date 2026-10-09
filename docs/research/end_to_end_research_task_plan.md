# 晶态无机材料端到端筛选技术路线

> 路线编号：RA（Research-grade Materials Screening）  
> 日期：2026-09-02  
> 状态：已确认；RA-0A、RA-1、RA-2、RA-3 工程验收完成，RA-0B 专业金标待审  
> 依赖基线：Screening Core、Workflow、MA-0～MA-5、Literature P0～P4B-lite、DA-0～DA-9.3  
> 第一版目标领域：晶态无机材料  
> 产品定位：多源证据驱动的材料候选筛选与科研决策支持平台

## 1. 路线调整决定

项目最初以无机半导体筛选为起点，现有能力已经包括 Materials Project 确定性查询与
排序、文献检索与全文证据、数据分析、统一页面、Master 路由、Artifact 和来源记录。
但现有 Agent 仍以“单次请求—单个结果”为中心，尚不能证明它们共同完成了一项可审计
的材料筛选研究。

旧版 RA 路线把“用户上传制备方法—硬度实验 CSV”作为唯一金标，导致 DataAnalysisAgent
成为主线，偏离材料筛选项目的核心。本版作出以下调整：

1. 删除“制备方法是否影响硬度”作为项目级金标；
2. 将第一版范围从“无机半导体”放宽为“晶态无机材料”，但只承诺已登记属性支持的任务；
3. MaterialsDatabaseAgent 负责候选生成和硬约束筛选，是主流程起点；
4. LiteratureAgent 负责实验合成、物相、实验性能、条件和局限核验；
5. DataAnalysisAgent 负责候选集质量、分布、跨来源数值对照和合法统计分析，不再要求
   每个任务都从用户实验 CSV 开始；
6. Master 升级为有状态筛选项目编排器，维护筛选合同、候选决策和证据缺口；
7. 先建立 Benchmark V1 和当前基线，再实施新合同与编排；
8. 严格按 RA-0～RA-6 顺序推进；在暂时没有材料专业审核人的条件下，RA-0 拆分为
   RA-0A 工程银标与 RA-0B 专业金标。RA-0A 通过后允许开发 RA-1～RA-3，RA-0B
   必须在进入 RA-4 科学综合和最终发布验收前完成。

目标流程调整为：

```text
应用需求与材料范围
  → 属性能力检查
  → 筛选合同与硬约束确认
  → 数据库候选生成
  → 确定性过滤和可解释排序
  → 重点候选文献核验
  → 跨来源身份与属性可比性检查
  → 支持、冲突、不可比较和证据缺口
  → 候选分级与验证建议
  → 可审计的科研筛选报告
```

## 2. 产品目标与完成定义

### 2.1 第一版产品目标

第一版完成的不是任意材料科研任务，而是一类边界明确的任务：

> 用户提出晶态无机材料的组成、结构、热力学或基础电子性质要求；系统在受支持的公开
> 数据源中产生候选，执行可复现的硬条件过滤和排序，对少量重点候选进行实验文献核验，
> 区分计算事实与实验事实，最终给出候选等级、证据、风险和下一步验证建议。

无机半导体仍是已支持的任务类型之一，但不再是顶层唯一领域。宽禁带氧化物、稳定相、
多晶型和亚稳相等晶态无机材料问题可通过同一框架接入。

### 2.2 一次任务的完成条件

一次筛选项目只有同时满足以下条件才能标记为 `completed`：

1. 用户确认应用目标、材料范围、硬约束、排序目标和候选数量；
2. 所有必需属性均通过能力检查，来源、单位、计算/实验角色明确；
3. 数据库查询保存为不可变快照，并具有稳定 query ID 和材料 ID；
4. 硬约束过滤结果可复现，最终候选中不存在已知硬条件违规；
5. 每个淘汰决定具有结构化原因，每个排序分数具有可解释组成；
6. 对进入深度核验的候选完成材料身份和实验文献证据检查；
7. 文献数值具有论文身份、页码或 chunk、原文依据、单位和条件；
8. 系统明确区分计算数据库、实验文献、用户数据和系统推断；
9. 主要结论链接到有效证据，或明确标记为证据不足；
10. 支持、冲突、不可比较和缺失证据没有被最终摘要隐藏；
11. 输出候选分级、不能说明什么和下一步最小验证方案；
12. Completion Gate 通过硬约束、身份、数值、来源、措辞和任务覆盖检查；
13. 生成普通材料科研人员可读的项目级 DOCX 和机器可审计底稿。

数据库、文献全文或关键属性不可用时，任务可以返回 `partial` 或 `unsupported`，但必须
列明已完成部分、阻塞项和可采取的下一步。证据不足时正确停止属于合格结果。

## 3. 第一版范围与非目标

### 3.1 包含范围

- 具有明确化学组成和晶体结构语义的无机材料；
- Materials Project 作为第一版主候选数据源；
- 组成、元素、化学体系、晶体结构、密度、金属/非金属、带隙、形成能、能量高于凸包
  等已支持属性；
- 通过属性注册表逐项开放的弹性、介电、磁性、声子、电极等扩展属性；
- 硬约束过滤、可解释排序、Top-K 和候选分级；
- 同化学式不同物相、多晶型、稳定相和亚稳相的身份处理；
- 最多 500 个进入项目快照的候选，最多 20 个进入快速文献核验，最多 5 个进入全文
  深度综合；
- 每个项目最多一次有界补充检索或返工；
- 支持、部分支持、冲突、不可比较和证据不足五类判断；
- 用户补充实验数据时，将其作为独立证据源接入；
- 项目级 DOCX，PDF 在 DOCX 视觉验收后转换；
- JSON/CSV/Markdown 作为审计和复现实验底稿。

### 3.2 不包含范围

- 聚合物、复合材料、生物材料和缺少明确晶体身份的样品；
- 把所有晶态无机材料属性都宣称为已支持；
- 仅凭数据库基础属性直接断言器件、催化、力学或生物性能优异；
- 自动预测数据库没有提供的关键性质；
- 自动提交 DFT、分子动力学、有限元或实验设备任务；
- 自动完成 XRD 精修、显微图像、光谱峰拟合或复杂机器学习；
- 将理想晶体自动等同于掺杂、缺陷、非化学计量、薄膜或纳米样品；
- 将数据库间计算一致性冒充实验验证；
- 绕过付费墙或自动下载无授权全文；
- 无限 Agent 协商、长期开放式自治或未经确认的外部写操作。

### 3.3 能力门原则

用户可以提出任意材料筛选问题，但系统必须先返回能力判断：

```text
supported：关键属性、来源和规则均已登记，可执行完整筛选
partial：只能完成部分过滤或文献调研，不能形成完整排序
unsupported：关键材料类型或核心属性不受支持
```

任何属性只有登记了定义、单位、来源、方法角色、缺失处理和可比性规则后，才能用于硬
约束或排序。界面不得静默使用近似属性替代用户要求。

## 4. Benchmark V1 与金标任务

### 4.1 测评包不是一份 CSV

Benchmark V1 由四类相互关联的固定资产组成：

```text
测试任务
├── 数据库查询快照
├── 文献证据包
├── 人工审核的标准答案
└── 自动评分规则与人工报告量表
```

同一套测评必须在改造前后重复运行，用于证明筛选准确性、证据质量、编排完整性和用户
可读性是否实际提高。不得用当前系统自身的一次输出直接生成标准答案。

### 4.2 数据规模

Benchmark V1 最低包含：

- 200～500 条冻结的 Materials Project 记录；
- 15 个确定性筛选与排序案例；
- 20 个材料身份、缺失值和跨来源边界案例；
- 8～15 篇允许使用且已经定位关键证据的文献；
- 30～50 条审核后的文献证据；
- 3 个端到端金标任务；
- 一份机器评分报告和一份非技术用户报告量表。

公开数据库数据属于源数据，不自动等于金标。确定性过滤答案需独立复核，材料身份、
文献可比性和最终科研判断需由相关专业人员批准。

### 4.3 三个端到端金标

#### 金标 A：稳定宽禁带氧化物候选筛选

目标：按用户确认的禁用元素、带隙、稳定性和能量高于凸包阈值筛选氧化物，核验重点
候选的实验物相和实验带隙。

重点验收：硬条件和 Top-K 正确；计算带隙与实验带隙不混用；仅满足带隙不被写成绝缘
强度或器件性能已获验证；文献方法和样品形态差异能够降低可比性。

#### 金标 B：同化学式不同晶型比较

目标：比较同一化学式的多个结构记录，判断稳定性、密度和基础电子性质差异，并用文献
核验实验中出现的物相。

重点验收：不按化学式错误合并不同空间群或结构；区分计算基态、亚稳晶型和实验物相；
身份不确定时停止精确属性合并；候选选择明确说明适用条件。

#### 金标 C：亚稳但具有实验合成证据的候选

目标：从能量略高于凸包的候选中，识别具有可靠实验合成证据的材料，并形成风险分级。

重点验收：`is_stable=False` 不被改写为“不能合成”；识别温度、压力、动力学或相变条件；
数据库判断与实验报道的张力进入冲突/限制；验证建议围绕物相、稳定窗口和复现条件。

### 4.4 测评层级

Benchmark 同时覆盖：

1. 数据库过滤与排序正确性；
2. 材料身份、物相和属性边界；
3. 文献证据抽取、定位和可比性；
4. 端到端编排、候选决策和报告；
5. 零结果、缺失属性、无全文、冲突和子 Agent 失败等反例。

## 5. 目标架构

```text
Unified Screening UI
        │
        ▼
ScreeningProjectService ───── ScreeningProjectStore
        │                              │
        ├── PropertyCapabilityRegistry │
        │                              ▼
        ▼                     Project State Machine
ScreeningOrchestrator
        │
        ├── MaterialsDatabaseAdapter ── MaterialsDatabaseAgent
        │         └── CandidateFilter / RankingService
        ├── LiteratureEvidenceAdapter ─ LiteratureAgent
        └── DataAnalysisAdapter ─────── DataAnalysisAgent
        │
        ▼
CandidateEvidenceLedger
        │
        ├── MaterialIdentityResolver
        ├── PropertyComparabilityEngine
        ├── ConflictAndGapEngine
        └── CandidateSynthesisService
        │
        ▼
ScreeningCompletionGate
        └── ScreeningReportService ── DOCX/PDF + JSON/CSV audit
```

### 5.1 新增组件职责

`ScreeningProjectService`

- 建立和修改筛选项目合同；
- 保存用户确认、范围变化和决策记录；
- 只通过稳定 ID 关联查询、材料、论文、分析和报告。

`PropertyCapabilityRegistry`

- 登记规范属性、单位、别名、来源和计算/实验角色；
- 声明属性能否用于查询、硬过滤、排序或仅用于背景；
- 声明缺失值、方法差异和单位转换规则；
- 在执行前产生 `supported/partial/unsupported` 能力判断。

`ScreeningOrchestrator`

- 将确认的筛选合同转换为有界工作项；
- 先数据库筛选，再只对重点候选调用文献核验；
- 校验每次结果并写入项目账本；
- 在证据缺口可安全补齐时执行最多一次补充任务；
- 不自行计算领域数值、不猜测材料映射、不替代子 Agent。

`CandidateEvidenceLedger`

- 保存候选、淘汰决定、排序依据、证据、可比性和冲突；
- 禁止自由文本成为唯一证据；
- 任何数值结论必须引用结构化来源或可定位原文。

`MaterialIdentityResolver`

- 规范化化学式、元素体系、材料 ID、空间群和结构指纹；
- 输出精确匹配、同组成不同相、相近体系或无法映射；
- 不只凭材料名称或化学式认定同一材料。

`CandidateSynthesisService`

- 综合硬约束、排序、实验支持、冲突和风险；
- 产生候选等级和逐项理由；
- 区分数据库计算、文献实验、用户实验与模型解释。

`ScreeningCompletionGate`

- 检查硬条件违规、材料身份、数值来源、文献定位、越界措辞和任务覆盖；
- Gate 未通过时返回具体缺口，不允许标记 `completed`。

## 6. 项目级核心合同

### 6.1 ScreeningProjectBrief

```python
class ScreeningProjectBrief:
    project_id: str
    title: str
    domain: Literal["crystalline_inorganic"]
    application_goal: str
    research_question: str
    material_scope: tuple[str, ...]
    required_elements: tuple[str, ...]
    excluded_elements: tuple[str, ...]
    allowed_material_classes: tuple[str, ...]
    hard_constraints: tuple["PropertyConstraint", ...]
    ranking_objectives: tuple["RankingObjective", ...]
    candidate_limit: int
    literature_review_limit: int
    allowed_sources: tuple[str, ...]
    user_confirmed: bool
```

`application_goal` 只是筛选背景，除非所需应用性能有直接证据，否则最终不得从基础属性
跨越推断真实应用表现。

### 6.2 属性合同

```python
class PropertyDefinition:
    property_id: str
    display_name: str
    canonical_unit: str | None
    aliases: tuple[str, ...]
    value_kind: Literal["number", "boolean", "category", "structure"]
    source_role: Literal["computed", "experimental", "mixed"]
    filterable: bool
    rankable: bool
    allowed_conversions: tuple[str, ...]
    comparability_requirements: tuple[str, ...]

class PropertyConstraint:
    property_id: str
    operator: Literal["eq", "in", "not_in", "min", "max", "between"]
    value: object
    unit: str | None
    missing_policy: Literal["exclude", "flag", "allow"]
    source_preference: Literal["computed", "experimental", "either"]

class RankingObjective:
    property_id: str
    direction: Literal["minimize", "maximize", "target"]
    target: float | None
    weight: float
```

硬约束和排序目标必须分离。排序权重只能在执行前确认；系统不得为了得到更顺眼的材料
而在运行后改变权重或阈值。

### 6.3 标准化候选与决定

```python
class MaterialCandidate:
    candidate_id: str
    project_id: str
    source_kind: str
    source_material_id: str
    formula: str
    chemsys: str
    structure_fingerprint: str | None
    space_group_number: int | None
    property_values: tuple["PropertyValue", ...]
    provenance: dict

class CandidateDecision:
    candidate_id: str
    hard_constraint_status: Literal["passed", "failed", "unknown"]
    exclusion_reasons: tuple[str, ...]
    ranking_score: float | None
    ranking_components: dict
    evidence_status: Literal[
        "supported", "partially_supported", "conflicted",
        "not_comparable", "insufficient"
    ]
    final_tier: Literal["A", "B", "C", "D", "unranked"]
    rationale: tuple[str, ...]
```

`A` 表示当前证据下优先验证，不表示材料已经被证明适用于目标应用。

### 6.4 工作计划、证据与状态

```python
class ScreeningWorkItem:
    work_id: str
    agent_name: Literal["materials_database", "literature", "data_analysis"]
    objective: str
    depends_on: tuple[str, ...]
    required_inputs: tuple[str, ...]
    expected_result_type: str
    status: Literal["pending", "running", "completed", "partial", "failed", "blocked"]
    attempt_count: int
    max_attempts: int

class ScreeningEvidence:
    evidence_id: str
    project_id: str
    candidate_id: str | None
    source_kind: Literal[
        "materials_database", "literature_full_text", "literature_abstract",
        "user_experiment", "data_analysis"
    ]
    source_id: str
    locator: dict
    property_id: str | None
    value: float | str | bool | None
    unit: str | None
    method: str | None
    conditions: dict
    source_excerpt: str | None
    review_status: Literal["validated", "human_reviewed", "pending", "rejected"]
    limitations: tuple[str, ...]

class ScreeningProjectState:
    project_id: str
    status: Literal[
        "draft", "capability_check", "needs_confirmation", "planned",
        "screening", "evidence_review", "synthesizing", "needs_user_input",
        "completed", "partial", "unsupported", "failed"
    ]
    brief_id: str
    plan_id: str | None
    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    unresolved_gaps: tuple[str, ...]
    revision: int
```

模型可起草结论语句，但状态、候选 ID、数值、单位和证据引用必须由确定性规则验证。状态
修改必须采用 revision 检查，避免旧回调覆盖较新状态。

## 7. Master 编排状态机

```text
draft
  → clarify_application_and_scope
  → check_property_capabilities
      ├── unsupported → explain_gap → unsupported
      └── supported / partial
  → confirm_screening_brief                 [人工门 1]
  → build_bounded_plan
  → confirm_material_assumptions_if_needed  [人工门 2]
  → query_materials_database
  → apply_hard_constraints
  → rank_candidates
  → select_literature_review_subset
  → run_literature_quick_review
  → run_selected_full_text_review
  → optional_candidate_data_analysis
  → normalize_identity_and_properties
  → assess_comparability
  → identify_support_conflicts_and_gaps
  → optional_single_follow_up
  → synthesize_candidate_decisions
  → review_shortlist                         [人工门 3]
  → completion_gate
  → create_screening_report
  → completed / partial
```

人工门 1 确认材料范围、硬约束、排序目标、属性来源偏好和候选数量。人工门 2 仅在物相
合并、属性近似或数据源选择会改变结果时出现。人工门 3 确认最终候选、关键缺口和报告
发布范围。界面不得逐字段机械提问。

失败与降级规则：

- 单个子 Agent 失败不清空已验证候选和证据；
- 主数据库失败时不得悄悄改用语义不同的数据源；
- 缺失关键属性时按项目约定排除、标记或降级，不得自动填充；
- 零结果时不得未经用户确认自动放宽硬约束；
- 文献无全文时只保留摘要背景；
- 材料映射不确定时保留缺口，不用相似材料冒充；
- 冲突证据不得通过多数投票消失；
- 项目恢复从最后已提交 revision 继续。

## 8. 三个子 Agent 的项目职责

### 8.1 MaterialsDatabaseAgent

负责候选生成、明确字段的查询过滤排序、材料详情、数据缺失和来源提示。它不得把计算
属性写成实验测量、猜测不支持属性、按名称相似自动映射或自行放宽约束。

### 8.2 LiteratureAgent

负责核验重点候选是否有实验合成或物相报道，提取样品、制备、测试条件和实验属性，识别
已知限制与冲突。必须区分摘要线索、全文证据、作者解释和系统综合；未取得全文时不得
产生精确实验条件和数值。

### 8.3 DataAnalysisAgent

负责候选快照质量、缺失模式、筛选前后分布、异常记录、可比较的计算值—实验值对照，
以及用户补充实验数据。不得对没有重复的数据强行推断显著性，不得合并不可比较来源，
不得事后改变阈值或权重。任务无需数值对照时可以跳过，并记录原因。

### 8.4 Master / ScreeningOrchestrator

负责项目状态、能力检查、计划、调用顺序、候选集缩减、证据规范化、缺口和最终综合。
不得自行生成领域数值、虚构文献内容、猜测材料映射或隐藏冲突。

## 9. 材料身份和跨来源可比性

### 9.1 材料身份层级

从强到弱：

1. 同一来源中的相同且有效的材料 ID；
2. 跨库结构匹配通过，且化学式、空间群和结构相容；
3. 精确化学式相同，但结构信息不完整；
4. 相同组成范围、掺杂母相或材料家族；
5. 仅共享元素体系；
6. 仅名称相似。

只有 1～2 可以认定为同一结构候选；3 必须标记身份不完全；4 属于相近材料；5～6 只能
用于背景检索。化学式相同不得覆盖多晶型差异。

特殊边界：掺杂材料不自动等同未掺杂母相；缺陷和非化学计量样品不自动映射到理想整数
化学式；薄膜、纳米颗粒、块体和单晶的性能比较必须检查形态；高压相、低温相和淬火
亚稳相必须保留条件。

### 9.2 属性可比性

每个属性比较至少检查定义、单位、计算/实验角色、计算泛函或测量方法、物相、样品形态、
温度、压力、气氛，以及数值是单值、区间、上下限还是定性结论。

可比性输出：

```text
comparable
partially_comparable
not_comparable
insufficient_metadata
```

同方向但不可比较的结果只能作为背景，不能提升候选置信度。

## 10. 候选综合规则

1. 先执行硬约束，再排序；硬约束失败的候选不得因文献热度重新进入 Top-K；
2. 硬约束未知与硬约束通过必须分开；
3. 排序分数必须展示组成，不能只返回黑箱总分；
4. 数据库计算性质与实验测量在每条结论中明确标注；
5. 跨数据库计算一致性不能替代实验验证；
6. 文献支持必须匹配材料身份、物相、指标和关键条件；
7. 方向相反且可比较的证据进入反证；
8. 证据不足不写成材料不具备该性质；
9. 亚稳不等于不可合成，已有合成报道也不等于常温长期稳定；
10. 仅基础属性满足要求不得直接声称目标应用性能优秀；
11. 机理只能来自可定位文献证据，或明确写成待验证假设；
12. 每个具体数值必须来自结构化证据；
13. 最终推荐使用“A/B/C/D 候选等级＋理由＋风险”，不得伪装绝对真值。

候选等级：A 为通过硬约束且有较强可比实验支持、优先验证；B 为通过硬约束但实验支持
有限；C 为存在可比冲突、稳定性或合成风险；D 为违反硬约束、身份不匹配或证据明确不
支持。A级仍只表示当前证据下优先验证。

## 11. 验证建议输出

`CandidateValidationPlan` 至少包括：要验证的候选和 Claim、当前证据缺口、组成与物相
确认、推荐表征、关键控制条件、数据库与文献冲突的判别实验或补充计算、最小对照材料、
成功/失败/停止标准和仍需专家决定的事项。

第一版不伪造精确合成参数、样本量或设备条件；缺少关键信息时必须列为研究缺口。

## 12. 用户界面

统一页面增加“材料筛选项目”视图。首轮依次确认用途或目标性质、材料范围、必须包含/
排除元素、硬条件、排序方式、候选数量和文献核验需求。

项目视图包括：

- 项目卡和属性能力状态；
- 工作计划；
- 候选漏斗和淘汰原因；
- 候选卡和排序依据；
- Candidate × 数据库 × 文献 × 用户数据证据矩阵；
- 独立的冲突与缺口；
- 推荐等级、边界和下一步；
- DOCX 报告及 JSON/CSV/Markdown 审计底稿。

界面一次只提出一个会改变候选集合或结论的问题。专家模式可展开材料 ID、查询 ID、
结构指纹、评分组成和证据定位。

## 13. 分阶段实施路线

### RA-0：冻结现状并建立 Benchmark V1

目标：先获得可重复的当前项目基线，解决“无法判断系统效果”的问题。

交付：

- 冻结现有 Master、三个子 Agent、筛选内核、DA-9.3 和跨 Agent 合同；
- 建立 200～500 条数据库快照、15 个筛选案例和 20 个边界案例；
- 建立 3 个端到端金标的任务合同、文献证据和人工标准答案；
- 定义候选、淘汰原因、排序、身份、来源、冲突和越界结论评分器；
- 用现有系统运行同一测评并记录失败类型；
- 建立 `ra0_crystalline_inorganic_benchmark_baseline.md`。

验收：数据来源、许可/访问条件、快照版本、标准答案产生方法、审核状态和评分规则可
审阅；现有全量离线测试通过。RA-0 不实现新编排器和新 UI。

阶段拆分：

- `RA-0A 工程银标`：冻结数据库快照、确定性答案、边界草案、带原文定位的文献银标、
  当前基线和自动测试。通过后可进入 RA-1～RA-3 的工程开发；
- `RA-0B 专业金标`：由材料、固体化学或晶体学专业人员审核物相身份、证据可比性、
  候选等级和验证建议。未通过前不得进入 RA-4 科学综合，不得将银标结果对外宣称为
  专业金标或发布级科研结论。

### RA-1：筛选合同、属性注册与能力门

目标：让系统只接受能够科学支持的筛选任务。

交付：筛选合同、属性别名/单位/来源角色/缺失策略/可比性规则、能力判断、项目存储、
revision、决策日志、状态转换和确认卡。

验收：未登记属性不能进入硬过滤或排序；缺失属性不被静默替代；非法状态转换、旧
revision 覆盖和未确认执行全部被拒绝。

### RA-2：候选、身份与 Claim–Evidence 账本

目标：建立跨数据库、文献和用户数据的一致项目对象。

交付：候选、决定、证据、Claim、三个只读适配器、材料身份规则、来源/定位/单位/方法/
条件校验、证据去重和不可变存储。

验收：同化学式不同晶型不会被错误合并；主要数值全部可追溯；数据库计算、全文实验、
摘要背景和用户数据不会混淆。

### RA-3：ScreeningOrchestrator 与候选漏斗

目标：把 Master 从单轮路由器升级为有界筛选项目编排器。

交付：计划和依赖、数据库候选生成、硬过滤、排序、重点候选选择、分层文献核验、失败
恢复、一次补充任务、项目进度和资源上限。

验收：三个金标形成可复现候选漏斗；硬约束违规候选为 0；调用有界；失败不丢证据；
重试不重复；Master 不直接生成材料数值。

### RA-4：可比性、冲突、候选分级与综合

目标：产生真正的跨来源候选决策，而不是结果拼接。

交付：身份和属性可比性、五类证据判断、计算/实验来源分区、排序稳健性、证据强度、
A/B/C/D 分级、综合服务和证据矩阵。

验收：金标支持、冲突和不可比较全部识别；不可比较数据不合并；亚稳不写成无法合成；
基础属性不越界推断应用性能；无来源具体数值为 0。

### RA-5：验证计划、项目报告与非技术界面

目标：交付普通材料科研人员可以直接审阅的完整筛选产物。

交付：验证计划、候选漏斗、候选卡、证据矩阵、风险和缺口、科研解释、中文 DOCX/PDF、
问答式入口、能力提示、项目进度和审计底稿。

验收：报告从目标贯穿到候选和验证计划；所有引用可定位；DOCX/PDF 逐页视觉验收通过；
普通用户无需阅读三个子 Agent 原始输出。

### RA-6：反例、跨任务泛化与最终发布门禁

目标：证明系统能正确筛选、正确拒绝和正确降级，并冻结晶态无机材料第一版。

至少覆盖：零候选、关键属性缺失、同化学式不同晶型、数据库计算冲突、计算与实验冲突、
亚稳材料已有合成证据、只有摘要、样品条件不可比较、掺杂材料错误映射、子 Agent 失败、
提示注入，以及聚合物/生物材料/复杂应用性能请求的能力门拒绝。

最终门禁：Ruff、mypy、全量离线测试、Benchmark 自动评分、金标人工评审、报告视觉检查、
来源覆盖率、硬约束正确率、身份判断、冲突识别和越界措辞拦截全部通过；建立
`crystalline_inorganic_screening_v1_baseline.md`。

## 14. 质量指标

| 指标 | 第一版门槛 |
|---|---:|
| 最终候选硬条件违规数 | 0 |
| 金标必保留候选召回率 | ≥95% |
| 淘汰原因正确率 | ≥95% |
| 确定性排序一致率 | 100% |
| 金标材料身份判断正确率 | 100% |
| 主要数值结构化来源覆盖率 | 100% |
| 文献关键证据可定位率 | 100% |
| 来源角色正确标注率 | 100% |
| 金标支持/冲突/不可比较识别率 | ≥90%，发布门禁目标 100% |
| 无证据具体事实率 | 0% |
| 基础属性到应用性能越界结论数 | 0 |
| 证据不足时错误放行数 | 0 |
| 已完成工作项重复执行率 | 0% |
| DOCX/PDF 阻断级视觉缺陷 | 0 |

报告另采用人工 1～5 分量表评价问题回答、候选理由、计算/实验区分、局限披露、下一步
可执行性和非计算机用户可读性。可读性不由报告长度替代。

## 15. 建议目录结构

```text
src/materials_screening/research/
  models.py
  project_store.py
  state_machine.py
  property_registry.py
  capability_gate.py
  work_plan.py
  orchestrator.py
  candidate_ledger.py
  material_identity.py
  comparability.py
  candidate_synthesis.py
  completion_gate.py
  validation_plan.py
  reporting.py
  adapters/
    materials_database.py
    literature.py
    data_analysis.py

tests/unit/research/
tests/integration/research/
tests/fixtures/research/benchmark_v1/
  database_snapshots/
  screening_cases/
  identity_cases/
  literature_evidence/
  end_to_end_tasks/
  expected/

docs/research/
```

领域子 Agent 保持在现有目录；`research/` 只负责项目级合同、候选账本和编排，不复制
数据库访问、文献检索或统计实现。

## 16. 数据来源演进

第一版使用 Materials Project 作为主候选和计算属性来源，LiteratureAgent 提供实验合成、
物相和实验性能，用户数据作为可选独立证据。

后续按独立阶段接入 COD、OQMD、JARVIS、NOMAD、Materials Cloud 和 OPTIMADE。跨库
适配器必须保留来源 ID、版本、方法和许可/访问说明；不得仅按化学式无条件合并，也不得
因为数据源公开就自动视为人工金标。

## 17. 安全、成本与可恢复性

- 外部文献、摘要、PDF 和数据库文本继续视为不可信数据；
- 项目计划不能扩大工具权限或执行未经确认的外部写操作；
- 所有工作项设记录数、文献数、重试和时间上限；
- 只对缩减后的候选做深度文献处理；
- 项目状态只存稳定引用，不把完整 PDF、大表或日志塞入会话；
- 查询快照和证据不可就地覆盖，修订生成新 revision；
- 报告只读取通过验证的 CandidateDecision、Claim 和 Evidence；
- 用户取消后停止新工作项，保留已完成结果；
- 真实网络失败不影响离线 Benchmark 门禁；
- 数据源许可和引用要求必须随快照记录。

## 18. 变更控制

1. 本文件重新确认前不开始 RA-0；
2. 确认后暂停继续堆叠单 Agent 功能，严格从 Benchmark V1 开始；
3. 每阶段只解决本阶段目标，不提前开发后续 UI、跨库适配或报告；
4. 新问题先登记，除非阻塞当前验收，不跨阶段插入实现；
5. 现有 Agent API 优先通过适配器复用，确需修改时保留兼容测试；
6. 模型只负责受约束草拟，数值、状态、来源、硬过滤和 Gate 由确定性程序控制；
7. 每阶段结束提交基线、测试证据、剩余缺口和唯一下一步；
8. 材料范围、硬约束、排序、数据源或结论修改必须进入决策日志；
9. 新增属性必须先进入 PropertyCapabilityRegistry 并补充金标；
10. 新增材料领域必须建立独立领域包和端到端金标，不自动继承支持声明。

## 19. 唯一下一步

RA-0A 工程银标以及 RA-1、RA-2、RA-3 已完成工程验收，RA-0B 专业金标保持待审且不得
被系统自行批准。下一步只完成 RA-0B 专业复核和签字；未通过前不进入 RA-4 科学综合，
不提前实现最终界面或跨数据库接入。
