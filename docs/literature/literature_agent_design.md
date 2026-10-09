# LiteratureAgent 技术设计与实施计划

> 执行说明（2026-08-20）：本文件保留原始技术架构与历史设计。后续实施阶段、当前
> 完成度和验收门统一以 `literature_agent_execution_baseline.md` 为唯一依据；本文开头
> 的 M0–M4 状态不再作为项目完成度判断。

> 状态：M0–M4 核心链路已实施并通过真实环境与单元测试；10 主题正式评测集仍待执行  
> 适用范围：`materials-screening-core` 多智能体系统  
> 目标子智能体：`literature` / `LiteratureAgent`  
> 当前状态：M0–M3 核心已实现；实验事实支持人工批准/驳回、不可变审计记录，且仅批准事实可生成 `kp_edges`。  
> M4 实验矩阵核心改造已完成：已新增 `experimental_groups`、`measurements`、`comparisons`、`paper_claims`、`claim_evidence_links`，并完成三篇真实论文验证、人工审核命令和严格结果组装。摘要结论视为待验证作者主张，正文表格与确定性比较作为证据层；新矩阵当前全部保持 `pending`，避免未经人工确认的数据进入知识图谱。  
> 上位约束：遵守 `TECHNICAL_SPEC.md`、`multi_agent_plan.md` 与现有
> `master/`、`agent/` 的安全和证据契约。

## 1. 目标与非目标

LiteratureAgent 接收检索主题、材料体系关键词及可选时间/语言条件，完成检索词
扩写、多源文献检索、去重排序、本地授权 PDF 摄取、RAG 检索、实验数据抽取和
证据约束的知识综合。对 Master 暴露为静态注册的
`delegate_to_literature(task)` 子智能体。

第一阶段输出遵循以下业务契约：

```json
{
  "papers": [
    {
      "title": "...",
      "doi": "10.xxxx/xxxx",
      "year": 2025,
      "key_findings": ["..."]
    }
  ],
  "synthesis_summary": "...",
  "data_tables": [],
  "kp_edges": []
}
```

非目标：绕过出版商权限下载全文；把 LLM 推断当成论文事实；执行任意 URL、Shell
或 Python；替代系统综述所需的双人筛选和偏倚评估；在首版构建通用知识图谱平台。

## 2. 大框架适配

### 2.1 依赖方向

```text
MasterAgent
  -> SubAgentRegistry（静态白名单）
    -> LiteratureAgent（独立 runner / conversation / checkpoint）
      -> literature/tools.py（AgentTool 适配层）
        -> literature/services/（确定性业务服务）
          -> literature/providers/（OpenAlex / S2 / PDF / embedding）
          -> literature/stores/（PostgreSQL + pgvector / 文件 artifact）
```

LiteratureAgent 可以复用 `agent/` 的模型、工具执行、策略、账本、最终校验和会话
设施，不得反向依赖 `agent_tools/`、`workflow/`、`planner/`，也不得直接修改其他
子智能体的数据。业务服务不依赖 Agent 上下文，便于离线单测和未来复用。

### 2.2 最小目录

```text
src/materials_screening/sub_agents/literature/
  __init__.py
  models.py
  prompt.py
  tools.py
  spec_factory.py
  services/
    query_expansion.py
    literature_search.py
    document_ingestion.py
    rag_retrieval.py
    experimental_extraction.py
    synthesis.py
  providers/
    openalex.py
    semantic_scholar.py
    pdf_parser.py
    embeddings.py
    reranker.py
  stores/
    literature_store.py
    artifact_store.py
  mock_model.py
```

基础设施实现放在 `services/literature/` 也可，但必须保持同一单向依赖原则。首版
建议放在子智能体包内，稳定后再提取公共服务。

## 3. 核心数据契约

所有边界模型使用 Pydantic v2，统一
`ConfigDict(extra="forbid", frozen=True)`；JSON 键稳定，时间为 UTC ISO 8601，
DOI 统一小写并移除 `https://doi.org/` 前缀。

### 3.1 请求

```python
class LiteratureRequest(BaseModel):
    topic: str                       # 1..1000 字符
    material_keywords: tuple[str, ...] = ()
    year_from: int | None = None
    year_to: int | None = None
    languages: tuple[str, ...] = ()
    max_papers: int = 20             # 1..100
    include_rag: bool = True
    extract_experimental_data: bool = False
```

LLM 扩写结果必须先落成 `ExpandedQuery`：包含原始词、规范化材料名、化学式、英文
同义词、缩写、过程/性能关键词和若干有界检索式。禁止让扩写器生成 URL 或 API
参数；服务层再把结构化字段映射到提供方参数。默认最多 6 个检索式，每个不超过
300 字符。

### 3.2 论文与证据

`PaperRecord` 至少包含：

- `paper_id`：内部稳定 ID；优先 DOI hash，其次 OpenAlex/S2 ID hash；
- `title`、`doi`、`publication_year`、`authors`、`venue`；
- `abstract`、`cited_by_count`、`open_access`、`landing_page_url`；
- `source_ids`：OpenAlex/S2 标识；
- `provenance`：来源、请求指纹、检索时间、原始记录 hash；
- `abstract_available` / `fulltext_available`，不得混淆摘要与全文证据。

`EvidenceSpan` 是综合与抽取的最小证据单元：`paper_id`、`chunk_id`、页码、字符
范围、原文片段 hash、可展示的短摘录、检索分数。所有 `key_findings`、数据单元格
与知识边都必须引用至少一个 `EvidenceSpan`。

### 3.3 最终输出

建议将用户给出的简化 Schema 扩为向后兼容的严格 Schema：

```text
LiteratureResult
  query_id: str
  papers: PaperSummary[]
    paper_id, title, doi?, year?, key_findings[], evidence_ids[]
  synthesis_summary: str
  synthesis_claims: SynthesisClaim[]
    claim, evidence_ids[], confidence
  data_tables: ExperimentalDataTable[]
  kp_edges: KnowledgeEdge[]
  warnings: str[]
  evidence_id: str
```

`KnowledgeEdge` 定义为 `subject - predicate -> object`，并带
`paper_id/evidence_ids/confidence/llm_extracted`。`kp_edges` 名称按需求保留，语义
固定为 knowledge-property edges，避免后续实现歧义。

实验表格每个数据行至少包含：材料、掺杂元素、掺杂量原值、规范化数值与单位、
制备条件、测试条件、性能指标、性能值与单位、证据页码、`llm_extracted=true`、
`review_status="pending"`、`extraction_model`、`prompt_version`。不能可靠转换单位时
保留原值并将规范化字段置空，禁止猜测。

## 4. 工具设计

所有工具满足现有 `AgentTool` Protocol，输入/输出严格校验，执行后通过
`context.id_generator.new_id()` 生成证据 ID，并调用 `context.ledger.record()`。

| 工具 | 作用 | 副作用分类 | 关键限制 |
| --- | --- | --- | --- |
| `openalex_search` | 免费元数据检索 | `READ_ONLY` | 超时、分页和总量上限；重建倒排摘要 |
| `s2_search` | 摘要、引用和补充元数据 | `READ_ONLY` | API key 可选；无 key 低限额；字段白名单 |
| `rag_retrieve` | pgvector 召回 + reranker 重排 | `READ_ONLY` | 仅检索已授权入库文档；返回证据片段 |
| `extract_experimental_data` | 从指定证据抽取数值表 | `READ_ONLY` | 只接受 `paper_id/chunk_id`，逐单元格溯源 |

PDF 摄取不应隐含在上述只读工具中。建议增加显式
`ingest_literature_documents`，输入仅允许工作区内文件路径或已验证的开放获取 URL
对象；副作用分类使用现有 `CREATE_WORKFLOW_RUN`（首版兼容方案）。中期应把
`ToolSideEffect` 扩为 `CREATE_KNOWLEDGE_ARTIFACT`，同时更新策略测试，避免语义
错配。用户未要求摄取时不得自动写库。

为了让一个子智能体在“检索→RAG→抽取→综合”间多步工作，LiteratureAgent 的
独立 `AgentSettings` 可启用 `agent_allow_multi_step_tools=True`，但仍限制每次模型
响应最多一个工具、每回合最多 8 次工具调用、至多 1 次摄取副作用。Master 仍保持
串行委派，不改变全局编排模型。

## 5. 检索、去重与排序

### 5.1 多源检索

1. 规范化用户主题和材料关键词；
2. LLM 生成结构化扩写，规则校验和截断；
3. 并发调用 OpenAlex 和 Semantic Scholar provider；
4. provider 只返回规范化 DTO，不把原始响应直接交给模型；
5. 合并、去重、排序并保存查询快照 `query_id`；
6. 后续 RAG、抽取、分页和综合复用快照，避免重复远程检索。

OpenAlex 摘要读取 `abstract_inverted_index` 并按位置确定性重建；无字段即明确标为
缺失。Semantic Scholar 使用显式 `fields`，401/403 不重试，429/5xx/超时进行带
jitter 的有限重试。客户端统一设置可识别的 `User-Agent`、连接/读取超时、每页和
总页数上限；日志不记录 API key 或完整响应。

### 5.2 去重

去重键优先级：规范化 DOI > provider ID 映射 > 规范化标题 + 年份。标题只用于
候选合并，采用保守阈值；发生作者/年份冲突时保留独立记录并产生 warning。字段
合并保留来源级 provenance，不用一个来源静默覆盖另一个来源。

### 5.3 首轮排序

采用确定性加权排序：文本相关性、材料关键词覆盖、摘要可用性、年份新近度与
`log1p(cited_by_count)`。引用数只能作为弱特征，不能压过主题相关性。排序权重、
tie-break 和版本写入查询快照；默认 tie-break 为 DOI、内部 `paper_id`。

## 6. PDF 摄取与 RAG

### 6.1 合法来源边界

仅摄取：用户上传/指定的本地 PDF、明确开放获取且下载许可可确认的 PDF、项目已
授权的数据集。检索结果中的 landing page 不能自动当作 PDF 下载地址。保存来源
URL、许可信息（若可得）、文件 SHA-256 与摄取时间。

### 6.2 解析与分块

PyMuPDF 提取页级文本和基础版面信息。先做页眉页脚去重、断词修复和空白规范化，
再按章节/段落分块；兜底使用 token 窗口。建议目标 700 tokens、重叠 100 tokens，
硬上限不超过 bge-m3 的 8192 token 输入。表格和图注单独成块，页码不可丢失。
扫描 PDF 首版返回 `OCR_REQUIRED`，OCR 作为后续可选能力，不静默产出空文本。

### 6.3 向量与索引

- 嵌入模型：`BAAI/bge-m3`，1024 维；模型 ID、revision、归一化方式入元数据；
- 存储：PostgreSQL 13+ 与 pgvector，`vector(1024)`；
- 索引：HNSW；首版 cosine distance，建库后用评测调节 `m`、
  `ef_construction`、查询 `ef_search`；
- 幂等键：`document_sha256 + chunker_version + embedding_model_revision +
  chunk_index`；
- 模型更换或分块版本变化时新建索引版本，不原位混写向量。

推荐表：`papers`、`paper_sources`、`documents`、`chunks`、`chunk_embeddings`、
`literature_queries`、`query_results`、`experimental_facts`、`knowledge_edges`、
`extraction_reviews`。数据库迁移使用显式 migration，不在工具执行期间自动改表。

### 6.4 召回与重排

`rag_retrieve` 先对 bge-m3 查询向量执行 HNSW top-N（默认 50），再用
`BAAI/bge-reranker-v2-m3` / `FlagReranker` 重排，返回 top-K（默认 8，最大
20）。检索必须支持 `paper_id`、年份和材料体系过滤。输出同时保留向量距离、
rerank 分数和最终顺序，分数不伪装成概率。

模型运行是可选重依赖：`rag` extra 单独声明，并通过 `EmbeddingProvider`、
`RerankerProvider` Protocol 注入。默认单元测试使用固定向量和固定重排 mock，不
下载 HuggingFace 权重、不需要 GPU、不访问网络。

## 7. 实验数据抽取与知识综合

### 7.1 抽取流程

1. 规则检索含掺杂量、温度、时间、性能单位的候选 chunk；
2. LLM 按严格 Schema 抽取，不允许自由文本表格；
3. 确定性解析数值、范围、误差和单位；
4. 对每个单元格校验原文支撑和页码；
5. 保存 `llm_extracted=true`、模型/提示词版本和 `pending` 复核状态；
6. 冲突值并存，禁止自动择一；未出现的数据保持 `null`。

只有 `review_status="approved"` 的事实可作为高置信知识边；`pending` 数据可返回给
用户，但必须显著标注“LLM 抽取，待人工复核”。人工修改保留原值、修改值、审阅
人和时间的审计记录。

### 7.2 综合约束

综合器只接收已规范化论文元数据和有界证据片段。每个可验证陈述绑定
`evidence_ids`；区分“论文报告”“跨论文一致趋势”“智能体推断”。相互矛盾的结果
必须并列呈现测试条件差异，证据不足时降级为不确定，不得补全 DOI、数值或结论。
最终回答沿用最近用户消息语言，论文题名、DOI、化学式、字段和单位保持原样。

## 8. 安全、可靠性与可观测性

- 仅允许预配置的 OpenAlex/S2 基础域名；拒绝重定向到非白名单域，防 SSRF；
- 本地 PDF 路径必须 resolve 后位于配置的 ingestion roots，限制文件大小、页数、
  MIME 和 PDF magic bytes；
- API key 使用 `SecretStr`/环境变量，只传 provider，不进入 state、artifact、日志；
- 远程请求使用连接/读取超时、有限重试、速率限制、断路和总结果上限；
- 原始外部文本视为不可信数据，提示词明确忽略论文中的指令性文本；
- artifact 采用稳定 JSON、SHA-256、临时文件 + 原子 replace、containment 和幂等；
- 工具输出设字节上限，较大论文列表只返回摘要页和 `query_id`；
- 观测指标：provider latency/error/rate-limit、去重率、摘要覆盖、摄取页数、chunk
  数、embedding/rerank 延迟、召回率、抽取复核通过率；不记录全文和密钥。

失败按稳定错误码分类：`INVALID_QUERY`、`PROVIDER_AUTH`、`PROVIDER_RATE_LIMIT`、
`PROVIDER_UNAVAILABLE`、`PDF_NOT_AUTHORIZED`、`PDF_PARSE_FAILED`、
`OCR_REQUIRED`、`VECTOR_STORE_UNAVAILABLE`、`MODEL_UNAVAILABLE`、
`EVIDENCE_INSUFFICIENT`。单一检索源失败时允许带 warning 的部分成功；所有来源
失败才整体失败。

## 9. 配置与依赖

建议新增可选依赖组，避免现有核心安装和离线 CI 被重模型拖累：

```toml
literature = ["PyMuPDF", "httpx", "psycopg[binary]"]
rag = ["sentence-transformers", "FlagEmbedding", "torch"]
```

实际版本在实施时锁定并通过 Python 3.11/3.12、CPU 环境与许可证检查后写入
`pyproject.toml`；不把未经验证的瞬时最新版写死在设计文档中。pgvector 由部署层
提供，开发/集成测试使用固定版本的 `pgvector/pgvector` 容器。

配置项建议：

```text
OPENALEX_BASE_URL / OPENALEX_MAILTO
S2_BASE_URL / S2_API_KEY
LITERATURE_HTTP_CONNECT_TIMEOUT / READ_TIMEOUT / MAX_PAGES
LITERATURE_INGEST_ROOTS / MAX_PDF_MB / MAX_PDF_PAGES
LITERATURE_DATABASE_URL
LITERATURE_EMBEDDING_MODEL / REVISION / DEVICE
LITERATURE_RERANKER_MODEL / REVISION / DEVICE
LITERATURE_RAG_TOP_N / TOP_K
```

## 10. 测试与验收

### 10.1 自动化测试

- Schema：extra 字段拒绝、边界长度、年份范围、DOI 规范化；
- provider：请求字段、分页上限、倒排摘要重建、超时/429/5xx 重试、401 不重试；
- 去重排序：跨源 DOI 合并、标题冲突、稳定 tie-break、provenance 保留；
- PDF：页码、页眉去重、分块重叠、扫描件识别、路径 containment、hash 幂等；
- RAG：1024 维检查、过滤、top-N/top-K、重排稳定性、索引版本隔离；
- 抽取：数值/范围/单位、空值、冲突、逐单元格证据、`llm_extracted`；
- Agent：四/五个工具白名单、ledger 登记、工具次数、副作用限制、证据 grounding；
- Master：注册、委派、结果包大小、子会话隔离和中文汇总；
- 安全：密钥泄漏、提示注入、SSRF、越界路径、超大文件和非 PDF；
- 离线门阻断 socket；真实 API、真实 PostgreSQL、真实模型均用独立 marker。

### 10.2 业务评测集

建立 10 个材料文献主题，每个主题保存：专家相关文献集合、查询变体、至少 5 篇
必须召回论文、关键结论证据、可抽取实验表及冲突案例。数据集需固定版本，不能在
评测时用实时搜索结果临时充当金标。

验收指标：

| 指标 | 门槛 | 计算方式 |
| --- | --- | --- |
| 主题召回 | 每主题相关文献 >= 5 篇 | top-20 去重结果与专家金标匹配 |
| Recall@20 | 建议同时报告 >= 0.70 | 10 主题宏平均 |
| 摘要忠实度 | 人工评分 >= 4/5 | 双人盲评，分歧仲裁 |
| 引文有效率 | 100% | 所有 key finding 可解析到证据片段 |
| 数值溯源率 | 100% | 每个非空抽取单元格有 paper/page/span |
| 离线回归 | 100% 通过 | 无网络、无数据库、无模型下载 |

忠实度量表：5=全部主张被证据直接支持；4=核心结论忠实，仅轻微措辞泛化；3=有
一处重要过度概括；2=多处不受支持；1=主要结论失真。除均分外，任何主题低于 3
均不得发布。

## 11. 分阶段实施计划

### M0：契约冻结与夹具（1–2 天）

- 冻结输入、输出、错误码、provider Protocol 与数据库迁移方案；
- 建立 OpenAlex/S2 脱敏 JSON fixture、3 类 PDF fixture 和 10 主题评测骨架；
- 产出架构决策记录：PDF 来源边界、`kp_edges` 语义、索引版本策略。

完成定义：Schema 单测通过；不接网络即可运行所有 fixture 测试。

### M1：元数据检索 MVP（3–5 天）

- 实现 OpenAlex/S2 provider、超时重试、规范化、去重排序和查询快照；
- 实现 `openalex_search`、`s2_search` 工具及证据账本；
- 实现 `models.py`、`prompt.py`、mock model、`spec_factory.py`；
- 注册到 Master，增加单 Agent/Master 集成测试。

完成定义：给定主题稳定返回至少 5 篇相关论文；部分源失败可降级；现有离线质量
门无回归。

### M2：PDF 摄取与 pgvector（4–7 天）

- 增加显式摄取工具、授权校验、PyMuPDF 解析、分块和 artifact；
- 增加 PostgreSQL migration、HNSW、幂等入库与索引版本；
- 接入 bge-m3 provider，CPU 冒烟测试与 mock 单测分离。

完成定义：同一 PDF 重复摄取不重复写入；每个 chunk 可追溯到文件 hash 和页码；
向量维度与模型版本一致。

### M3：重排、抽取与综合（4–6 天）

- 接入 bge-reranker-v2-m3，实现 `rag_retrieve`；
- 实现候选 chunk 选择、严格数值抽取、单位处理、复核状态；
- 实现 evidence-grounded synthesis 与最终 `LiteratureResult`；
- 增加冲突论文、缺摘要、缺全文、扫描 PDF 等降级路径。

完成定义：所有结论和数值通过证据校验；LLM 抽取数据均标记待复核。

### M4：评测、硬化与交付（3–5 天）

- 完成 10 主题金标、Recall@20、每主题命中数和双人忠实度评测；
- 压测分页、限流、长 PDF、并发查询和工具输出大小；
- 完成部署说明、数据迁移/备份、模型缓存和故障排查文档；
- 运行 `ruff format/check`、`mypy src`、离线 pytest 及独立真实门。

完成定义：达到用户给定验收指标，安全测试无高危问题，Master 可稳定委派并返回
严格 Schema。

## 12. 发布顺序与风险

建议先发布“元数据检索 MVP”，再以 feature flag 开启 PDF/RAG/抽取。主要风险及
控制如下：

| 风险 | 控制 |
| --- | --- |
| S2 无 key 限流 | provider 限速、缓存、可选 key、OpenAlex 降级 |
| 开放获取状态不等于可自动下载 | 显式许可/来源校验，默认不下载 |
| CPU 嵌入与重排延迟高 | 批处理、模型常驻、可配置 device、异步摄取 |
| PDF 表格解析不稳定 | 页级证据、原值保留、人工复核；后续再加专用表格解析 |
| 跨论文条件不一致导致错误综合 | 强制记录实验/测试条件，冲突并列，不做无依据归一 |
| 向量模型升级污染索引 | revision 固定、版本化索引、离线重建后原子切换 |
| 工具副作用枚举语义不足 | 首版兼容映射，中期新增知识 artifact 类型并补策略测试 |

## 13. 实施前需冻结的三个决策

1. 首版 PDF 是否只接受本地/用户上传文件；建议是，开放获取自动下载延后。
2. PostgreSQL 是复用现有部署还是 LiteratureAgent 独立实例/schema；建议独立
   schema 与最小权限账号。
3. 实验抽取首版支持的指标与单位白名单；建议先选 2–3 类材料任务（例如光催化
   降解率、产氢速率、带隙），用评测驱动扩展，避免一开始做无边界通用抽取。
