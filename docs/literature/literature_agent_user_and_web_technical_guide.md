# LiteratureAgent 技术、测试与可视化网页接入指南

> 文档版本：1.0  
> 日期：2026-08-21  
> 当前里程碑：P1–P3、P4A、P4B-lite 已完成；P5 十主题正式评测尚未执行。

## 1. 产品目标与可信边界

LiteratureAgent 帮助用户发现材料领域论文，把用户有权访问的 PDF 转换为可审核的单篇
结构化证据，并在用户明确选文后组装多篇报告。系统采用“候选—审核—正式结果”三层
数据流，优先保证可追溯性，而不是追求无人值守的自动结论。

可信边界如下：

- 在线元数据和摘要用于发现与初筛，不能代替全文实验事实。
- PDF 必须由用户合法提供，系统不绕过登录、验证码或付费墙。
- LLM 抽取结果默认 `pending`，必须保留页码、chunk、逐字引文和文本哈希。
- 不可靠翻译、无逐字证据、组归属歧义、图中估读和缺失单位不得自动批准。
- 正式报告和 `kp_edges` 只读取 `approved` 数据。
- 多篇报告的论文集合由用户明确指定，系统不自动纳入检索结果。

## 2. 已实现架构

```text
CLI / 未来 Web API
  ├─ 主题检索
  │    ├─ DeterministicQueryExpander
  │    ├─ OpenAlexProvider
  │    ├─ SemanticScholarProvider
  │    └─ LiteratureSearchService → 查询快照
  ├─ PDF 单篇分析
  │    ├─ PyMuPDF → chunks
  │    ├─ bge-m3 → pgvector/HNSW
  │    ├─ AutomatedDossierExtractor → 待审论文档案
  │    └─ AutomatedMatrixExtractor → 待审实验矩阵
  ├─ P4B-lite 批次编排
  │    └─ 每篇隔离、断点复用、失败隔离、批次 JSON
  └─ 用户选择与正式组装
       └─ LiteratureIntegrationService → LiteratureResult
```

主要实现位置：

| 模块 | 位置 | 职责 |
| --- | --- | --- |
| CLI | `src/materials_screening/cli.py` | 用户命令和显示层 |
| 数据模型 | `sub_agents/literature/models.py` | 严格输入输出 Schema |
| 检索编排 | `sub_agents/literature/service.py` | 扩写、多源合并、排序、快照 |
| Provider | `sub_agents/literature/providers.py` | OpenAlex/S2 请求与规范化 |
| PDF/RAG | `sub_agents/literature/rag.py` | 解析、向量化、检索、重排 |
| 单篇档案 | `sub_agents/literature/automation.py` | 证据约束档案抽取和安全翻译 |
| 实验矩阵 | `sub_agents/literature/matrix_automation.py` | 组、测量与主张抽取 |
| 数据库 | `sub_agents/literature/pgvector_store.py` | 文档、chunk、矩阵、审核事件 |
| 批次状态 | `sub_agents/literature/batch.py` | P4B-lite 批次持久化 |
| 正式整合 | `sub_agents/literature/integration.py` | 已审核证据组装 |

## 3. 数据与状态

### 3.1 文件数据

| 路径 | 内容 |
| --- | --- |
| `data/literature_queries/<query_id>` | 可复现检索快照 |
| `data/literature_dossier_candidates` | 待审核单篇档案 |
| `data/literature_dossiers` | 已审核单篇档案 |
| `data/literature_dossier_reviews` | 不可变档案审核事件 |
| `data/literature_batches/<batch_id>.json` | 批次汇总 |
| `data/literature_pdfs` | 默认授权 PDF 目录 |

### 3.2 PostgreSQL/pgvector 数据

核心表包括文档、文本 chunk、实验组、测量、比较、论文主张、主张—证据链接和不可变
矩阵审核事件。向量维度固定为 bge-m3 的 1024 维；HNSW 用于近邻候选检索，重排器只
改变候选顺序，不修改证据内容。

### 3.3 关键标识

- `query_id`：检索请求、扩写版本、Provider 集和排序版本的稳定指纹。
- `document_id`：PDF 内容 SHA-256 前缀；文件改名不改变 ID。
- `batch_id`：排序后的批次 document ID 集合指纹。
- `group_id`、`measurement_id`：实验实体稳定 ID。

前端和 API 应使用这些 ID，不应把标题或文件名作为主键。

## 4. 端到端实际验收方案

### 4.1 测试材料

选择一个未参与开发的主题，通过 P4A 找到 5–10 篇候选，从中合法下载 2–3 篇文本型
研究论文。第一轮避免扫描 PDF、只有摘要的文件和缺少正文的补充材料。

### 4.2 验收步骤

1. 运行 `literature search`，保存 `query_id`，检查扩写词、年份和候选相关性。
2. 检查每篇候选是否具有题名、DOI或稳定入口、年份和 Provider 状态。
3. 下载 2–3 篇并运行 `batch-analyze`。
4. 确认批次中的每篇论文具有独立 `document_id`；单篇错误不影响其他论文。
5. 查看 `dossier-pending`，核对研究问题、材料、方法、结果、机理、限制和复现信息。
6. 查看 `matrix-show --summary`，核对实验组、指标、数值、单位和原文组归属。
7. 重复运行同一批次，确认显示复用已有档案/矩阵，而不是重新抽取。
8. 仅批准准确论文；指定其中 2 篇运行 `result-build`。
9. 确认未指定论文未进入报告，所有正式事实可回到页码和逐字引文。

建议记录：检索相关性（1–5）、档案忠实度（1–5）、档案完整度（1–5）、数值准确率、
单位准确率、需人工纠正条数、单篇处理时间和失败原因。

### 4.3 通过标准

- 无需针对新论文修改代码或编写专用 staging JSON。
- 至少 2 篇完成 PDF → 待审核档案/矩阵。
- 批次重跑能够复用结果。
- 错误、缺表或来源限流均明确展示，不静默伪造。
- 用户选文后的报告不包含未指定论文。
- 已批准事实的页码、原文和文档 ID 可解析率为 100%。

P5 才执行固定 10 主题 Recall@20 与专家相关文献集评测；一次用户验收不能替代 P5。

## 5. 当前命令契约

| 用户动作 | CLI |
| --- | --- |
| 主题检索 | `literature search <topic>` |
| 批量逐篇分析 | `literature batch-analyze --pdf ...` |
| 查看批次 | `literature batch-show <batch_id>` |
| 查看待审档案 | `literature dossier-pending <document_id>` |
| 审核档案 | `literature dossier-review <document_id> ...` |
| 查看实验矩阵 | `literature matrix-show <document_id> --summary` |
| 审核矩阵 | `literature matrix-review <document_id> ...` |
| 生成选文报告 | `literature result-build --document-id ...` |
| 环境诊断 | `literature doctor` |

网页后端不得通过拼接 Shell 字符串调用这些命令。应直接调用对应 Pydantic 模型、Service
和 Store；CLI 只作为当前人工测试适配层。

## 6. 可视化网页建议

### 6.1 页面结构

建议先做 5 个页面，不做复杂知识图谱编辑器：

1. **主题检索页**：主题、材料关键词、年份、数量；显示扩写词、Provider 状态和候选卡片。
2. **上传与批次页**：拖拽 PDF、逐篇进度、失败重试、复用标记和 `batch_id`。
3. **单篇论文页**：左侧档案栏目，右侧页码/逐字证据；风险筛选与缺失栏目提示。
4. **实验矩阵页**：实验组为行、指标为列；点击单元格查看原文；审核操作必须二次确认。
5. **选文与报告页**：用户勾选已审核论文，设置主题问题，生成报告并查看证据来源。

### 6.2 后端 API 草案

```text
POST /api/literature/search
GET  /api/literature/queries/{query_id}
POST /api/literature/batches
GET  /api/literature/batches/{batch_id}
GET  /api/literature/documents/{document_id}/dossier
GET  /api/literature/documents/{document_id}/matrix
POST /api/literature/documents/{document_id}/dossier-review
POST /api/literature/documents/{document_id}/matrix-review
POST /api/literature/reports
GET  /api/literature/reports/{report_id}
```

长任务应返回 `202 Accepted + job_id`，前端轮询或使用 SSE 获取阶段状态。不要让 HTTP 请求
一直等待模型、向量化和全文抽取完成。

建议任务状态：

```text
queued → parsing → indexing → dossier_extracting → matrix_extracting
       → pending_review | completed | partial | failed
```

### 6.3 前端展示规则

- 明确区分“元数据”“摘要主张”“正文证据”“人工批准事实”。
- `pending` 使用黄色，`approved` 使用绿色，`rejected` 使用灰/红色；颜色之外还要有文字。
- 摘要和原文默认折叠，点击后展示，不在列表页堆积全文。
- 每个数值显示指标、数值、单位、实验组、页码和原文入口。
- Provider 限流显示为降级，不显示成检索失败或双源成功。
- 跨论文比较必须显示“可比”“部分可比”“不可直接比较”，不能只画统一柱状图。

### 6.4 安全约束

- 上传文件只允许 PDF，限制数量、单文件大小和页数，并重新验证 MIME/文件头。
- 服务器生成文件名；禁止使用用户文件名拼接路径。
- PDF 只进入配置的 ingestion root；防止路径穿越和任意 URL 抓取。
- API key、数据库密码仅在服务端环境变量中使用，绝不下发浏览器。
- 审核操作记录 reviewer、时间、旧状态、决定和原因，禁止无审计覆盖。
- 报告 API 只接收当前用户有权访问且已审核的 document ID。

### 6.5 推荐实施顺序

1. 先建立只读 API：检索、批次查看、档案查看、矩阵查看。
2. 再增加 PDF 上传和异步任务队列。
3. 再增加审核 API 和身份认证。
4. 最后增加选文报告、导出和有限图表。

不要先做漂亮页面再补证据状态，也不要让前端直接读取 PostgreSQL 表。

## 7. 已知限制与后续决策

- Semantic Scholar 无 key 时容易限流；OpenAlex 是当前可靠降级来源。
- PyMuPDF 适合文本型 PDF，复杂表格、扫描件和补充材料可能不完整；MinerU 暂作为未来
  可选解析器，不影响当前主流程。
- 首次加载 bge-m3/reranker 较慢且占用数 GB 模型缓存。
- 当前批次顺序执行，适合测试和小批量；网页化后应使用有并发上限的后台任务队列。
- 现有多篇报告基于已批准证据组装，但研究问题导向的高级叙事与跨论文可比性模型仍可
  在真实使用数据充分后迭代，不应现在凭三篇金标过度设计。

阶段状态和历史验收以
[`literature_agent_execution_baseline.md`](literature_agent_execution_baseline.md) 为准。

