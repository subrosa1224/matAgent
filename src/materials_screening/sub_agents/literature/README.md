# LiteratureAgent

LiteratureAgent 是材料文献检索、授权 PDF 分析和证据约束知识整合子智能体。当前推荐的
用户流程是：

```text
主题检索 → 用户选择并合法获取 PDF → 批量逐篇分析
         → 用户查看/审核单篇结果 → 用户指定论文 → 生成多篇报告
```

系统不会绕过付费墙，不会把摘要当作全文实验事实，也不会自动决定哪些论文应当合并。

## 当前能力

- P4A：OpenAlex + Semantic Scholar 统一检索、确定性中英文查询扩写、跨源去重排序、
  年份/材料筛选、查询快照和单源失败降级。
- PDF RAG：PyMuPDF 解析、bge-m3 向量化、PostgreSQL + pgvector/HNSW、本地检索与
  bge-reranker-v2-m3 可选重排。
- 单篇分析：中文论文档案、逐字证据、页码、风险等级、实验组、测量、比较和摘要主张
  核验。
- P4B-lite：最多 20 篇 PDF 的逐篇隔离分析、断点复用、失败隔离和批次状态汇总。
- 人工审核：候选默认为 `pending`；只有 `approved` 证据可进入正式报告和知识关系。

当前不实现自动下载付费全文、图表 OCR 数值猜测、自动选文或无需人工审核的全自动
跨论文结论。

## 准备环境

在项目根目录安装依赖：

```powershell
uv sync --extra workflow --extra literature --extra rag
```

`.env` 至少配置：

```dotenv
LITERATURE_DATABASE_URL=postgresql://matagent:你的密码@127.0.0.1:5432/matagent
LITERATURE_INGEST_ROOTS=D:\Desktop\matAgent\data\literature_pdfs
LLM_PROVIDER=intern
INTERN_API_KEY=你的令牌
```

可选配置包括 `S2_API_KEY`、`OPENALEX_MAILTO`、`LITERATURE_EXTRACTION_MODEL`、
`LITERATURE_MODEL_DEVICE` 和模型 revision。不要把 `.env` 或令牌提交到版本库。

启动并检查数据库：

```powershell
$env:MATAGENT_POSTGRES_PASSWORD = "与 .env 数据库 URL 一致的密码"
docker compose -f deploy\literature\compose.yaml up -d
uv run materials-screen literature migrate --yes
uv run materials-screen literature doctor
```

## 实际测试流程

### 1. 检索主题

```powershell
uv run materials-screen literature search `
  "多孔磷酸钙生物陶瓷骨诱导结构孔隙率" `
  --year-from 2020 `
  --max-papers 10 `
  --sort balanced `
  --show-abstract
```

`--sort` 支持 `balanced`（默认）、`recent` 和 `relevance`。结果分为核心相关、高度相关
和扩展阅读；严格主题可能返回少于上限的论文，系统不会用明显不相关记录凑数。

需要机器可读结果时使用 `--json`。Semantic Scholar 被限流时，OpenAlex 结果仍会返回，
同时明确显示来源降级。

### 2. 准备 PDF

从出版社、学校数据库或合法开放获取入口下载 2–3 篇未参与开发的文本型 PDF，放入
`data\literature_pdfs`。第一轮不要选扫描件，也不要一次测试大量论文。

### 3. 先做批量快速阅读

```powershell
uv run materials-screen literature batch-preview `
  --topic "你的研究主题" `
  --pdf "data\literature_pdfs\paper-1.pdf" `
  --pdf "data\literature_pdfs\paper-2.pdf"
```

每篇只进行一次轻量模型调用，输出研究问题、方法、主要结论、相关性和是否建议深度分析；
不生成实验矩阵，也不要求审核。相同 PDF 与主题的预览会直接复用。
若模型连续两次未返回合格中文，系统会生成保守的安全中文回退卡片，不虚构主要结论。
检索数据源不可用时，可复用同主题最近成功快照并用当前相关性规则重新排序，输出会
明确标记 `STALE_SNAPSHOT_FALLBACK`。

### 4. 只深度分析用户选中的论文

```powershell
uv run materials-screen literature batch-analyze `
  --pdf "data\literature_pdfs\paper-1.pdf" `
  --pdf "data\literature_pdfs\paper-2.pdf"
```

该普通模式默认只生成论文档案，不运行耗时且需要高级审核的实验矩阵。确需结构化实验
组、测量和比较时，由管理员追加 `--extract-matrix`。

也可使用 `--directory data\literature_pdfs`。命令返回 `batch_id` 和每篇稳定的
`document_id`。整批新 PDF 只加载一次向量模型；重新运行会复用已有结果。已有实验组但
测量为 0 的论文只有在显式增加 `--retry-incomplete` 时才会重试矩阵抽取。

### 5. 查看单篇结果

```powershell
uv run materials-screen literature dossier-pending <document_id>
uv run materials-screen literature matrix-show <document_id> --summary
uv run materials-screen literature matrix-status <document_id>
```

已批准档案使用：

```powershell
uv run materials-screen literature dossier-show <document_id>
```

### 6. 审核

档案和矩阵应在核对原文后分别审核：

```powershell
uv run materials-screen literature dossier-review <document_id> `
  --decision approved --reviewer "你的名字" --yes

uv run materials-screen literature matrix-review <document_id> `
  --decision approved --reviewer "你的名字" --yes
```

如果存在错误，不要整篇批准；先保留 `pending` 并记录问题。

### 7. 用户指定论文生成报告

```powershell
uv run materials-screen literature result-build `
  --document-id <document_id_1> `
  --document-id <document_id_2> `
  --summary
```

只有命令中明确指定且已经审核的论文证据会进入正式报告。

## 常用诊断

```powershell
docker compose -f deploy\literature\compose.yaml ps
uv run materials-screen literature doctor
uv run materials-screen literature search --help
uv run materials-screen literature batch-analyze --help
```

详细架构、数据契约、验收记录及网页设计见
[技术与测试指南](../../../../docs/literature/literature_agent_user_and_web_technical_guide.md)。
