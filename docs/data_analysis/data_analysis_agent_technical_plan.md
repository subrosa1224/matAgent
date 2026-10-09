# 数据分析 Agent 技术路线与实施计划书

> 文档状态：v1.0 已完成（DA-0～DA-7 全部验收）  
> 制定日期：2026-08-31  
> 适用项目：`materials-screening-core` / `matAgent`  
> 目标子 Agent：`data_analysis`  
> 依赖基线：统一多智能体 MA-0～MA-5 已验收  
> 实施原则：后续工作严格按本文 DA-0～DA-7 的顺序、边界和验收门槛推进；未完成当前阶段验收不得进入下一阶段。

## 1. 背景与定位

项目当前已具备 `materials_database`、`literature`、Master 路由、通用 Artifact、
结果渲染注册表和统一 Gradio 页面。数据库 Agent 能对 Materials Project 查询快照
进行描述统计、比较和离群检测，但它不应承担用户实验表格的通用清洗、统计推断、
可视化和分析报告工作。

本计划新增独立的 DataAnalysisAgent，其领域边界为：

1. 分析用户上传的 CSV/JSON 表格；
2. 分析受信任的材料数据库查询快照和筛选 Workflow 结果；
3. 执行确定性的数据质量检查、清洗、描述统计、相关性和统计检验；
4. 生成可追溯的图表、清洗数据和 Markdown/CSV/JSON 报告；
5. 通过 Master 与数据库、文献 Agent 串行协作；
6. 不让模型直接计算统计量或执行任意 Python 代码。

## 2. 交付目标

完成 DA-0～DA-7 后，用户可以在统一页面中：

- 上传 UTF-8 CSV 或 JSON 数据集；
- 查看数据规模、字段类型、预览、缺失率、重复行和疑似异常值；
- 对指定字段执行描述统计、分组统计和 Pearson/Spearman 相关分析；
- 在前提检查后执行受支持的组间统计检验并获得效应量和置信区间；
- 以显式、可审计的规则清洗数据，并保留不可变原始数据；
- 生成直方图、箱线图、散点图、折线图、柱状图和相关热力图；
- 导出 Markdown 报告、CSV/JSON 结果与清洗后的数据；
- 在连续对话中引用“刚才的数据”“上一次分析”和稳定的 Artifact/analysis ID；
- 由 Master 在数据分析、材料数据库和文献 Agent 之间进行有界路由与汇总。

## 3. 范围与非目标

### 3.1 本轮 MVP 范围

- 输入格式：`.csv`、顶层为对象数组的 `.json`；
- 数据规模：默认最大 50 MB、100,000 行、500 列；
- 字段类型：数值、布尔、字符串/类别、可解析日期；
- 数据质量：缺失、重复、类型混杂、常量列、唯一值数量、数值范围和 IQR 离群提示；
- 清洗：选择列、过滤行、去重、类型转换、缺失值删除/常量填补/均值或中位数填补；
- 描述分析：总体和分组统计、分位数、相关矩阵；
- 统计检验：独立样本 t 检验、Welch t 检验、配对 t 检验、Mann–Whitney U、
  单因素 ANOVA、Kruskal–Wallis；
- 统计报告：样本量、前提检查、统计量、p 值、效应量、置信区间、警告；
- 可视化：直方图、箱线图、散点图、折线图、柱状图、相关热力图；
- 导出：Markdown、CSV、JSON、PNG；
- 接入：SubAgentRegistry、Master 自动路由、统一 UI、Artifact 和 Renderer Registry；
- 质量保证：单元、集成、路由、安全、确定性和端到端验收测试。

### 3.2 明确不纳入本轮

- Excel/XLSX、Parquet、数据库直连和远程 URL 下载；
- OCR、PDF 表格抽取和图片表格识别；
- 任意 Python、SQL、Shell、Notebook 或用户代码执行；
- 自动训练预测模型；
- 线性/逻辑回归、PCA、聚类、特征重要性和超参数搜索；
- 因果推断、贝叶斯分析、生存分析和时间序列预测；
- 自动改写原始数据或静默删除样本；
- 用 LLM 生成或修正数值分析结果；
- 子 Agent 直接互相调用；
- 公开部署、账号权限、多租户和长任务队列。

以上能力若需加入，必须先修改本文、说明数据契约和验收标准，再进入实施。

## 4. 强制架构原则

### 4.1 分层

```text
Unified UI
  └─ Unified Application Controller / ArtifactRegistry
       └─ MasterAgentRunner
            └─ DataAnalysisAgent
                 └─ Whitelisted Agent Tools
                      └─ DataAnalysisService
                           ├─ DatasetStore
                           ├─ deterministic statistics
                           └─ Report / Plot exporters
```

- UI 只负责上传、会话状态、流式状态和展示；
- Master 只负责任务识别、串行编排和结果汇总；
- DataAnalysisAgent 只负责领域任务规划、工具选择和基于工具证据解释；
- Tool 层负责 Pydantic 输入校验、证据登记和副作用声明；
- Service 层负责全部数值计算、文件读写和结果持久化；
- 原始数据集不可变，清洗操作必须产生新的 `dataset_id`；
- 大型表格不进入 LLM 上下文，只传摘要、分页预览和稳定引用。

### 4.2 禁止的实现

- 在 UI 中用领域关键词分支替代 Master 自动路由；
- 将本地绝对路径传给模型或显示给普通用户；
- 把完整数据集序列化进提示词、会话状态或结果信封；
- 使用 `eval`、`exec`、动态导入或子进程执行分析；
- 允许模型提供任意绘图表达式、过滤表达式或 Python 代码；
- 在统计前提不满足时静默选择或替换检验；
- 只报告 p 值而不返回样本量、效应量、方法和警告；
- 在未经用户显式请求时执行有数据变换副作用的工具；
- 修改数据库或文献 Agent 的内部实现来适配数据分析 Agent；
- 将上传文件名直接作为存储路径；
- 复用现有面向 `MaterialRecord` 的 `DataFileParser` 解析通用实验表格。

## 5. 目标代码结构

```text
src/materials_screening/
├─ data_analysis/
│  ├─ __init__.py
│  ├─ models.py                 # 数据集、分析、图表和报告领域契约
│  ├─ dataset_store.py          # 私有路径与不可变版本存储
│  ├─ parser.py                 # 通用 CSV/JSON 表格解析
│  ├─ service.py                # 确定性分析与清洗服务
│  ├─ statistics.py             # 统计量、检验、效应量和前提检查
│  └─ reporting.py              # Markdown/CSV/JSON/PNG 导出
└─ sub_agents/
   └─ data_analysis/
      ├─ __init__.py
      ├─ models.py              # 工具输入/输出模型
      ├─ tools.py               # 白名单 AgentTool
      ├─ prompt.py              # 领域提示词
      ├─ deterministic_model.py # 首工具确定性路由约束
      ├─ mock_model.py          # 离线验收模型
      ├─ spec_factory.py        # SubAgentSpec 工厂
      └─ README.md

tests/
├─ fixtures/data_analysis/
├─ unit/data_analysis/
├─ unit/sub_agents/test_data_analysis_*.py
├─ unit/master/test_data_analysis_routing.py
└─ integration/test_data_analysis_agent.py

docs/data_analysis/
├─ data_analysis_agent_technical_plan.md
├─ da0_contract_baseline.md
├─ da1_dataset_baseline.md
├─ da2_analysis_baseline.md
├─ da3_mutation_baseline.md
├─ da4_presentation_baseline.md
├─ da5_agent_baseline.md
├─ da6_application_baseline.md
└─ da7_acceptance_report.md
```

根据实现中的内聚性可以合并小型模块，但不得打破分层和依赖方向。

## 6. 核心数据契约

### 6.1 数据集引用

```json
{
  "dataset_id": "dataset-<opaque-id>",
  "source_artifact_id": "artifact-data-<opaque-id>",
  "display_name": "experiment.csv",
  "format": "csv",
  "row_count": 120,
  "column_count": 8,
  "schema_fingerprint": "sha256:...",
  "content_fingerprint": "sha256:...",
  "parent_dataset_id": null,
  "created_by_operation_id": null
}
```

- `dataset_id` 和 Artifact ID 必须是不透明安全 ID；
- 路径只保存在 `DatasetStore` 私有映射中；
- `content_fingerprint` 用于重复文件识别；
- 清洗结果通过 `parent_dataset_id` 形成不可变派生链。

### 6.2 字段画像

```json
{
  "name": "conductivity",
  "inferred_type": "numeric",
  "non_null_count": 116,
  "missing_count": 4,
  "unique_count": 109,
  "sample_values": [1.2, 1.5, 1.8],
  "warnings": []
}
```

样例值数量必须有界并经过 JSON 安全转换；不得在错误中泄漏整行敏感内容。

### 6.3 分析结果

```json
{
  "analysis_id": "analysis-<opaque-id>",
  "dataset_id": "dataset-<opaque-id>",
  "analysis_type": "descriptive",
  "method": "describe",
  "parameters": {},
  "summary": {},
  "warnings": [],
  "evidence_id": "evidence-<opaque-id>"
}
```

结果必须包含实际采用的方法和参数。统计检验额外包含各组样本量、前提检查、
统计量、自由度（适用时）、p 值、效应量、置信区间及多重检验校正信息。

### 6.4 派生操作记录

```json
{
  "operation_id": "operation-<opaque-id>",
  "source_dataset_id": "dataset-...",
  "result_dataset_id": "dataset-...",
  "operations": [
    {"kind": "drop_duplicates", "parameters": {"subset": ["sample_id"]}}
  ],
  "rows_before": 120,
  "rows_after": 118,
  "warnings": []
}
```

清洗工具必须先校验全部操作，再一次性产生新数据集；任何失败不得留下半成品。

### 6.5 图表和报告 Artifact

- `dataset_file`：原始或派生 CSV/JSON；
- `analysis_result`：结构化分析 JSON；
- `analysis_plot`：PNG；
- `analysis_report`：Markdown；
- owner 固定为 `data_analysis`；
- Unified UI 只显示 Artifact ID、显示名、类型和状态，不显示内部路径。

## 7. 白名单工具设计

### 7.1 `inspect_dataset`

输入：`dataset_id`、可选预览偏移和条数。  
输出：数据集摘要、字段画像、有界预览、质量警告。  
副作用：只读。

### 7.2 `assess_data_quality`

输入：`dataset_id`、可选字段列表、离群提示阈值。  
输出：缺失、重复、常量列、混合类型、IQR 提示和严重级别。  
副作用：只读。

### 7.3 `describe_dataset`

输入：`dataset_id`、数值字段、可选 `group_by`、分位数。  
输出：计数、均值、标准差、最小值、分位数、最大值和分组结果。  
副作用：只读。

### 7.4 `analyze_correlations`

输入：`dataset_id`、字段列表、`pearson|spearman`、缺失值策略。  
输出：相关矩阵、成对样本量、强相关摘要和常量字段警告。  
副作用：只读。

### 7.5 `run_statistical_test`

输入：`dataset_id`、响应字段、分组字段或配对字段、检验方法、显著性水平。  
输出：检验结果、前提检查、效应量、置信区间、可解释警告。  
副作用：只读。

工具不允许 `auto` 在多个检验中静默择优。若提供 `recommended` 模式，必须返回推荐理由，
并由后续显式工具调用执行具体检验。

### 7.6 `transform_dataset`

输入：`dataset_id`、有序的结构化操作列表。  
输出：新的 DatasetReference、操作记录和前后规模。  
副作用：创建派生数据集；必须由用户明确要求清洗或转换。

### 7.7 `create_analysis_plot`

输入：`dataset_id`、固定图表类型、字段、分组、标题和轴标签。  
输出：图表 Artifact、作图数据规模和警告。  
副作用：创建 PNG Artifact。

不得接受任意代码、任意文件名、任意路径或任意 matplotlib 参数字典。

### 7.8 `create_analysis_report`

输入：`dataset_id`、分析 ID 列表、图表 Artifact ID 列表、格式。  
输出：报告 Artifact 和结构化目录。  
副作用：创建 Markdown/JSON/CSV 文件；仅在用户明确要求导出或生成报告时执行。

## 8. 统计方法与解释规范

### 8.1 确定性

- 所有统计计算由固定版本的 Python 库和项目代码执行；
- 同一数据、方法和参数必须产生相同结构化结果；
- 浮点值在存储时保留计算精度，展示层统一格式化；
- 随机算法不在 MVP 范围内；
- 缺失值处理策略必须写入结果参数。

### 8.2 方法边界

- Pearson：只用于数值字段，报告成对有效样本数；
- Spearman：对秩相关进行确定性处理；
- Student t：必须报告方差齐性检查结果；
- Welch t：不要求方差齐性；
- 配对 t：必须有明确配对键或等长、顺序已验证的配对数据；
- ANOVA：报告组数、每组样本量和前提警告；
- 非参数检验：说明比较的是分布/秩差异，避免误写为均值差异；
- 小样本、常量列、零方差、空组、严重缺失或检验不适用时返回结构化错误，
  不产生伪结果。

### 8.3 解释模板

最终回答至少覆盖：

1. 使用了什么数据和字段；
2. 使用了什么方法及原因；
3. 样本量和关键统计量；
4. 差异或关系的方向与大小；
5. p 值、效应量和置信区间（适用时）；
6. 前提、缺失值策略和警告；
7. 结论边界，不把相关性表述为因果性。

## 9. 文件、资源与安全限制

- 只从 ArtifactRegistry 或 DatasetStore 解析文件，不接受模型提供的本地路径；
- 上传文件按内容哈希和安全扩展名存储；
- CSV 使用 `utf-8-sig`，拒绝无法解码的文件，不静默猜测系统编码；
- JSON 顶层必须是对象数组，嵌套对象在 MVP 中拒绝或作为字符串处理；
- 文件、行、列、单元格长度、预览行数和工具输出大小全部有上限；
- CSV 导出防止公式注入：以 `= + - @` 开头的文本单元格进行安全处理；
- PNG/报告路径由服务生成，禁止路径穿越和覆盖任意文件；
- 普通用户回答不显示 API Key、内部路径、异常栈和完整原始行；
- 数据变换和导出均登记 ToolLedger 与 evidence ID；
- 原始数据不可覆盖或删除。

## 10. 依赖策略

当前项目已有 `pandas`。本 MVP 的统计检验和 PNG 绘图新增独立可选依赖组：

```toml
analysis = [
  "scipy>=1.15,<2",
  "matplotlib>=3.10,<4",
]
```

- 不在 MVP 中加入 scikit-learn、seaborn、statsmodels 或 notebook 运行时；
- 核心数据集检查与描述统计保持只依赖 pandas；
- 缺少 `analysis` 依赖时，基础工具仍可导入，统计检验/绘图返回明确的能力不可用错误；
- 锁文件更新和依赖安装必须在 DA-2/DA-4 对应阶段完成并验收。

## 11. Master、UI 与跨 Agent 路由

### 11.1 单 Agent 路由

- 上传 CSV/JSON 后请求清洗、统计、检验或绘图 → `data_analysis`；
- Materials Project 查询、筛选、详情、数据库快照统计 → `materials_database`；
- PDF 阅读、论文检索、论文事实提取 → `literature`；
- 只有“数据库查询结果需要进一步通用统计分析”时才进行数据库到数据分析的跨 Agent 计划。

路由不能只依赖“分析”一词，因为“分析论文”和“分析数据库候选”仍可能属于已有 Agent。

### 11.2 附件所有权

- `.csv`/`.json` 上传 Artifact owner 为 `data_analysis`；
- `.pdf` owner 为 `literature`；
- Artifact 类型是附件路由的强信号；
- 自动模式由 Master 结合用户意图和 Artifact owner 决定；
- 显式模式仍通过 `SubAgentUiPluginRegistry` 注册，不在主分发器增加领域长分支。

### 11.3 有界跨 Agent 流程

MVP 只允许下列串行流程：

```text
MaterialsDatabaseAgent
  → query/result Artifact validation
  → DataAnalysisAgent
  → Master synthesis
```

```text
LiteratureAgent
  → structured experiment-matrix Artifact validation
  → DataAnalysisAgent
  → Master synthesis
```

```text
DataAnalysisAgent
  → material identifiers/structured conditions validation
  → MaterialsDatabaseAgent
  → Master synthesis
```

- 每轮最多调用三个领域子 Agent；
- DataAnalysisAgent 不直接调用其他 Agent；
- 跨 Agent 只传稳定 ID 和严格结构化契约，不传内部路径或大段自由文本；
- 文献提取值与数据库值必须保留来源边界，不得混写为同一测量体系。

## 12. 分阶段实施路线

### DA-0：基线冻结与契约

目标：先冻结范围和公共契约，不实现分析业务。

实施项：

- 新增本文；
- 新增数据分析领域核心 Pydantic 契约；
- 新增 Artifact 类型和结果类型的契约测试；
- 明确与现有 `DataFileParser`、MaterialsDatabaseAgent 的边界；
- 建立 `docs/data_analysis/da0_contract_baseline.md`。

验收：

- 新契约 `extra="forbid"`、有界、可 JSON 往返；
- 非法 ID、额外字段、重复引用和不一致状态被拒绝；
- 不修改既有子 Agent 业务；
- Ruff、mypy、DA-0 单测和现有应用契约定向回归全部通过。

### DA-1：安全数据集与质量画像

目标：建立通用表格的安全导入、不可变存储和只读检查能力。

实施项：

- `DatasetStore`、CSV/JSON parser、内容哈希和路径隔离；
- 数据规模限制、类型推断、分页预览；
- 缺失、重复、常量列、唯一值和 IQR 质量画像；
- 固定 fixtures 和确定性单测；
- 建立 `da1_dataset_baseline.md`。

验收：

- 正常 CSV、BOM CSV、JSON 可稳定导入；
- 空文件、坏 JSON、嵌套 JSON、超限数据和路径穿越被拒绝；
- 原始文件不被修改；
- 大数据不进入模型上下文；
- Ruff、mypy、DA-1 单测和 DA-0 回归全部通过。

### DA-2：描述统计、相关性与统计检验

目标：完成 MVP 的全部只读数值分析。

实施项：

- 总体/分组描述统计；
- Pearson/Spearman 相关性及成对样本量；
- 六类统计检验、前提检查、效应量和置信区间；
- `analysis` 可选依赖中的 SciPy；
- 与手算小样本和 SciPy 基准对照测试；
- 建立 `da2_analysis_baseline.md`。

验收：

- 固定数据结果与参考值在声明误差范围内一致；
- NaN、Inf、常量列、空组、小样本和非法方法有明确行为；
- 结果包含方法、参数、样本量和警告；
- Ruff、mypy、DA-2 单测和 DA-0～DA-1 回归全部通过。

### DA-3：不可变清洗与派生数据

目标：实现有审计记录的数据清洗，不覆盖原始数据。

实施项：

- 结构化清洗操作白名单；
- 操作预校验和原子写入；
- 数据集父子链、操作记录和清洗后质量复查；
- CSV 公式注入防护；
- 建立 `da3_mutation_baseline.md`。

验收：

- 每次清洗生成新 `dataset_id`；
- 原始哈希和文件保持不变；
- 失败操作不留下半成品；
- 行列变化与操作记录一致；
- Ruff、mypy、DA-3 单测和前序回归全部通过。

### DA-4：可视化、报告与导出

目标：生成可复用的 PNG 和结构化分析报告。

实施项：

- 六类固定图表；
- Matplotlib 无界面后端和资源释放；
- Markdown、CSV、JSON 报告导出；
- Renderer 与 Artifact 元数据；
- 图像尺寸、标签、空数据和中文字体降级测试；
- 建立 `da4_presentation_baseline.md`。

验收：

- 图表可打开、尺寸有界、轴和图例符合输入；
- 导出不覆盖任意路径且不存在 CSV 公式注入；
- 报告引用稳定的 dataset/analysis/artifact ID；
- 无图形资源泄漏；
- Ruff、mypy、DA-4 单测和前序回归全部通过。

### DA-5：DataAnalysisAgent 与白名单工具

目标：把确定性能力封装成独立 SubAgentSpec。

实施项：

- 工具输入/输出模型和八个白名单工具；
- ToolLedger/evidence 登记和副作用声明；
- 专用 system prompt；
- 确定性首工具约束、离线 mock 模型和 runner factory；
- 独立 README 和 Agent 级测试；
- 建立 `da5_agent_baseline.md`。

验收：

- Agent 不执行任意代码，不访问未登记路径；
- 查询与变换工具的副作用策略正确；
- LLM 回答中的数值可追溯到工具结果；
- mock 模式不访问网络且可完成核心场景；
- Ruff、mypy、Agent 单测和前序回归全部通过。

### DA-6：Master、Artifact 与统一 UI 接入

目标：以注册方式接入统一多 Agent 应用。

实施项：

- 注册 `data_analysis` SubAgentSpec；
- Master 自动路由和 mock 路由；
- CSV/JSON Artifact 注册与安全复制；
- `SubAgentUiSpec + handler` 显式模式；
- `analysis_answer` Renderer；
- 上传、停止、错误恢复、连续会话和专家审计测试；
- 建立 `da6_application_baseline.md`。

验收：

- 普通数据分析请求不会误路由到数据库或文献；
- PDF 路由无回归；
- 普通用户界面不显示内部路径；
- 自动和显式模式均可完成数据检查；
- 既有 `master ask`、数据库和文献页面定向回归通过；
- Ruff、mypy、UI/Master 测试和前序回归全部通过。

### DA-7：跨 Agent、端到端验收与文档

目标：完成有界协作、整体回归和发布基线。

实施项：

- 数据库查询快照 → 数据分析的结构化适配；
- 文献实验矩阵 → 数据分析的结构化适配；
- 数据分析结果 → Master 来源分区综合；
- 端到端验收数据集与失败场景；
- README、使用指南、限制说明和最终验收报告；
- 建立 `da7_acceptance_report.md`。

验收：

- 三条允许的跨 Agent 流程至少各有一个端到端测试；
- 文献、数据库、用户文件来源严格分区；
- 失败和部分成功能恢复并保留已完成结果；
- 全量定向测试、Ruff、mypy 通过；
- 所有 DA-0～DA-7 基线文档与实现一致。

## 13. 阶段门禁与变更控制

每个阶段严格执行：

```text
实现 → 单元测试 → Ruff → mypy → 定向回归 → 基线记录 → 进入下一阶段
```

规则：

1. 当前阶段存在失败项时，不开始下一阶段；
2. 修复不得通过删除测试、放宽断言或吞掉异常实现；
3. 若发现计划缺项，先修改本文并记录原因，再修改代码；
4. 新增依赖、工具、Artifact 类型或跨 Agent 流程必须更新本文；
5. 每阶段只修改该阶段所需文件；
6. 对当前工作区已有未提交改动，采用最小补丁，不覆盖或回退用户改动；
7. 每阶段建立独立基线 Markdown，记录命令、测试数量、已知限制和结论；
8. 任何暂缓项必须写入基线，不得口头略过；
9. DA-7 前不宣称 DataAnalysisAgent 已完整交付。

## 14. 测试矩阵

| 类别 | 必测内容 |
|---|---|
| 契约 | 严格字段、边界、ID、JSON 往返、不可变性 |
| Parser | BOM、中文、空值、引号、坏 JSON、超限、嵌套对象 |
| Store | 哈希去重、路径隔离、父子链、并发、原子写入 |
| 质量 | 缺失、重复、常量、混合类型、IQR、空数据 |
| 统计 | 参考值、NaN、Inf、小样本、零方差、多组、配对错误 |
| 清洗 | 预校验、操作顺序、回滚、不可变原始数据、公式注入 |
| 绘图 | 六类图、空数据、标签、资源释放、文件安全 |
| 工具 | 输入拒绝、side effect、ledger、输出大小、证据 ID |
| Agent | 工具选择、连续对话、拒绝任意代码、mock/real 边界 |
| Master | 自动路由、显式路由、歧义词、附件所有权、调用上限 |
| UI | 上传、停止、恢复、路径隐藏、Artifact 显示、渲染 |
| 跨 Agent | 来源分区、结构化适配、部分失败、综合结论边界 |

## 15. 验收场景

### 场景 A：数据质量

上传含缺失、重复和非法数值的 CSV，询问：

```text
检查这份数据有什么质量问题，不要修改数据。
```

预期：路由到 DataAnalysisAgent，只执行检查工具，原文件不变，返回字段级问题和证据。

### 场景 B：分组统计

```text
按烧结温度分组，比较电导率的均值、中位数和标准差。
```

预期：返回每组样本量与统计表，说明缺失值策略，不自动执行显著性检验。

### 场景 C：显著性检验

```text
比较 A、B 两种制备方法的硬度是否存在显著差异，并报告效应量。
```

预期：使用用户指定或明确说明的检验，返回前提、样本量、p 值、效应量、区间和边界。

### 场景 D：显式清洗

```text
删除 sample_id 重复项，用各组中位数填补 conductivity 缺失值，保存新数据。
```

预期：生成派生数据集和操作记录，原数据哈希不变。

### 场景 E：图表与报告

```text
画出温度与电导率的散点图，并把刚才的统计结果生成 Markdown 报告。
```

预期：生成 PNG 和 Markdown Artifact，回答中无内部路径。

### 场景 F：数据库到分析

```text
把刚才数据库候选的带隙和形成能做相关分析。
```

预期：通过结构化查询快照适配，不重新编造或复制数据库记录。

### 场景 G：文献矩阵到分析

```text
比较这些论文中不同烧结温度下的抗压强度。
```

预期：只使用带文献来源的结构化实验矩阵，结论保留论文间条件不可比警告。

## 16. 完成定义

DataAnalysisAgent 只有同时满足以下条件才算完成：

- DA-0～DA-7 全部阶段通过；
- 所有计划内工具有严格契约、确定性服务和测试；
- 原始数据不可变，所有派生操作可审计；
- 自动路由、显式模式、Artifact 和 Renderer 均完成注册；
- 统计结果能追溯到 dataset/analysis/evidence ID；
- 普通用户界面无内部路径和原始异常泄漏；
- 数据库与文献 Agent 既有行为无回归；
- 最终验收报告记录测试命令、结果、限制和未纳入功能；
- README 和使用指南与实际行为一致。

## 17. 后续增强候选（不属于本轮）

完成 DA-7 后可另立计划评估：

- XLSX/Parquet；
- 回归与置信区间诊断；
- PCA、聚类和可解释特征工程；
- 实验设计与下一批实验条件推荐；
- 大数据异步任务和任务队列；
- 交互式 Plotly 图表；
- 多租户数据隔离与保留策略。

这些能力不得在本轮实现中提前夹带。
