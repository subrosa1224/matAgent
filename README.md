# materials-screening-core

无机半导体材料多智能体筛选系统。

## 架构

```
WebUI / CLI
    │
    ▼
┌─────────────────────────────────┐
│         Master Agent            │  意图识别 → 委派 → 汇总
│         src/materials_screening/master/
└──────────┬──────────────────────┘
           │ delegate_to_<name>(task)
    ┌──────┼──────┬──────┐
    ▼      ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ 材料  │ │ 文献  │ │ 数据  │ │ 计算  │   子 Agent，各占独立文件夹
│ 筛选  │ │ 检索  │ │ 分析  │ │ 任务  │   src/materials_screening/sub_agents/<name>/
└──────┘ └──────┘ └──────┘ └──────┘
```

- **Master** — 接收用户自然语言，识别意图，委派给对应子 Agent，汇总结果
- **子 Agent** — 各自独立的 LangGraph Agent，拥有专属工具集和 system prompt
- **基座模型** — 书生 Intern-S2-Preview-35B

## 快速开始

```bash
uv sync --extra workflow

# 配置 API — 复制并编辑 .env
cp .env.example .env

# 单 Agent 模式
uv run python -m materials_screening agent ask -m "寻找带隙>2eV的稳定氧化物"

# 多智能体模式
uv run python -m materials_screening master ask -m "寻找带隙>2eV的稳定氧化物" --progress

# WebUI
uv run python -m materials_screening agent ui
```

## 构建与测试

```bash
uv run ruff format . && uv run ruff check .
uv run mypy src

# 离线质量门
uv run pytest -m "not real_api and not real_llm and not real_agent" -q
```

## 配置大模型

在 `.env` 中列出所有可用 API，`LLM_PROVIDER` 一行切换：

```bash
LLM_PROVIDER=intern

INTERN_BASE_URL=https://chat.intern-ai.org.cn/api/v1/
INTERN_MODEL=intern-s2-preview-35b
INTERN_API_KEY=YOUR_API_TOKEN
INTERN_THINKING_MODE=true

CUSTOM_BASE_URL=http://localhost:8000/v1
CUSTOM_MODEL=my-model
CUSTOM_API_KEY=
```

Intern 模型请求默认直连，不更改全局代理；连接失败或超时默认最多尝试 2 次，不重复执行 Agent 工具。需要继承系统代理时可设置 `INTERN_USE_SYSTEM_PROXY=true`。详见 [连接策略](docs/research/intern_transport_policy.md)。

## Materials Database Agent

Master Agent 现在优先通过 `materials_database` 子 Agent 访问 Materials
Project。该 Agent 提供以下白名单能力：

- 条件筛选、排序与 Top-K；
- 材料详情和多材料比较；
- 显式请求的描述统计与离群检测；
- 显式请求的 CSV、JSON 或 Markdown 导出。

每次查询会生成可复用的 `query_id`。后续分页、比较、统计、离群检测和
导出都读取同一查询快照，不重复访问 Materials Project。查询快照保存在
`data/material_queries/`。未要求统计、离群检测或导出时，Agent 不会自动
执行这些操作。

旧的 `materials_screening` 和 `outlier_detection` Master 委派入口暂时保留为
兼容别名，实际由 `materials_database` 处理。

## Data Analysis Agent

统一页面支持上传 UTF-8 CSV、顶层对象数组 JSON 或 XLSX（读取首个工作表），并自动
路由到 `data_analysis`。上传后会立即执行只读的一键 EDA；无需写提示词，也可在数据集
面板中选择字段并点击操作按钮。该 Agent 提供：

- 数据结构检查、质量画像、描述统计和 Pearson/Spearman 相关性；
- 六类显式统计检验、效应量和置信区间；
- 不覆盖原始数据的白名单清洗；
- 六类固定 PNG 图表、面向科研人员的 Word 报告，以及 Markdown/CSV/JSON 审计和
  安全导出。

仓库内置三套可直接上传的演示数据，分别用于测试质量检查、分组统计/绘图和材料属性
分析，见 [`examples/data_analysis/README.md`](examples/data_analysis/README.md)。

每个上传文件先由 ArtifactRegistry 校验和安全复制，再登记为不可变
`dataset_id`。Agent 只接收稳定 ID，不接收本地路径，也不执行任意 Python、
SQL、Shell 或 Notebook 代码。数据库查询快照和人工审核过的文献实验矩阵可通过
结构化适配进入数据分析；不同来源在 Master 汇总时保持独立分区。

安装统计和绘图依赖并启动统一页面：

```powershell
uv sync --extra workflow --extra web-ui --extra analysis
uv run materials-screen master ui
```

完整用法和限制见
[`docs/data_analysis/data_analysis_agent_usage.md`](docs/data_analysis/data_analysis_agent_usage.md)。

## Literature Agent

LiteratureAgent 当前已完成 P4A 与 P4B-lite，可执行以下闭环：

```text
自然语言主题检索
  → 用户选择并合法获取 PDF
  → 批量中文快速预览
  → 只深度分析选中的论文
  → 查看并审核单篇论文档案
  → 用户指定论文生成多论文证据对照报告
```

检索使用 OpenAlex 与 Semantic Scholar，支持中英文查询扩写、跨源去重、相关性分层、
年份筛选、`balanced`/`recent`/`relevance` 排序和单源失败降级。PDF 分析使用
PyMuPDF、bge-m3、PostgreSQL + pgvector/HNSW 及可选 bge-reranker。系统会复用已经
索引的 PDF、预览、论文档案和实验矩阵，不会在每次运行时重新加载模型。

PDF 正式题名优先从已核实书目元数据或 PDF 内部/第一页提取，最后才回退到文件名。
报告按栏目归组证据，同一小标题只显示一次，并保留各证据页码；多论文报告使用正式
论文题名进行对照。

系统不会绕过付费墙、自动下载付费全文、把摘要当作全文实验事实，或用明显不相关论文
凑足数量。查询快照保存在 `data/literature_queries/<query_id>/result.json`，批次状态
保存在 `data/literature_batches/<batch_id>.json`。

用户显式提供的材料体系会作为检索硬约束。当前数据源限流或失效且无法形成合格候选
时，系统会寻找同一扩写概念的最近成功快照，并用当前规则重新过滤排序；输出会带有
`STALE_SNAPSHOT_FALLBACK` 提示，不会把历史结果伪装为实时结果。

### LiteratureAgent 环境准备

推荐使用固定的 Compose 配置启动数据库：

```powershell
$env:MATAGENT_POSTGRES_PASSWORD = "请设置强密码"
docker compose -f deploy\literature\compose.yaml up -d
uv sync --extra workflow --extra literature --extra rag
uv run materials-screen literature migrate --yes
uv run materials-screen literature doctor
```

`doctor` 是只读诊断；`migrate` 才会显式创建 `vector` 扩展、文档表、chunk表和
HNSW索引。数据库未配置或依赖缺失时，PDF RAG工具不会注册，但OpenAlex元数据
检索仍可使用。

Windows + Docker Desktop 建议在 `LITERATURE_DATABASE_URL` 中使用
`127.0.0.1` 而非 `localhost`，避免本机IPv6/名称解析回退造成连接延迟。

`.env` 至少配置：

```dotenv
LITERATURE_DATABASE_URL=postgresql://matagent:数据库密码@127.0.0.1:5432/matagent
LITERATURE_INGEST_ROOTS=D:\Desktop\matAgent\data\literature_pdfs
LLM_PROVIDER=intern
INTERN_API_KEY=你的令牌
```

`S2_API_KEY` 为可选项，用于提高 Semantic Scholar 限额。Compose 展开
`${MATAGENT_POSTGRES_PASSWORD}` 时读取的是当前 PowerShell 环境或 Compose 的环境文件；
项目应用读取的 `.env` 不一定会自动成为 Compose 环境变量，因此启动数据库前建议显式
设置 `$env:MATAGENT_POSTGRES_PASSWORD`。

### LiteratureAgent 实际使用

普通用户也可以直接启动本地可视化工作台：

```powershell
uv sync --extra web-ui
uv run materials-screen literature ui --port 8503
```

浏览器访问 `http://127.0.0.1:8503`。页面采用与数据库子智能体一致的问答式交互：
用户直接描述检索主题，或上传PDF后提出“快速预览”“深度分析第1、3篇”“综合这些
论文”等请求，LiteratureAgent 会根据当前对话上下文调用对应能力。上传的有效PDF会
安全复制到配置的本地文献目录。当前文献界面独立运行，但保留统一对话入口，便于后续
与主智能体合并；网页仅监听本机地址，不会自动公开到互联网。
若指定端口已被占用，程序会自动尝试后续端口，并在终端显示实际访问地址。

1. 检索候选论文：

```powershell
uv run materials-screen literature search `
  "多孔生物陶瓷孔结构与成骨性能" `
  --year-from 2020 `
  --max-papers 10 `
  --sort balanced `
  --show-abstract
```

2. 用户合法下载选中的 PDF，并放入 `data\literature_pdfs`。系统当前不代替用户获取
   受限全文。

3. 批量快速预览。该步骤输出每篇论文的中文研究问题、方法、主要结论、主题相关性和
   深度分析建议，不生成实验矩阵，也不需要审批：

```powershell
uv run materials-screen literature batch-preview `
  --topic "多孔生物陶瓷孔结构与成骨性能" `
  --pdf "data\literature_pdfs\paper-1.pdf" `
  --pdf "data\literature_pdfs\paper-2.pdf"
```

非法 JSON 或非中文预览会自动重试一次；`CCK-8`、`3D` 等标准术语中的数字会保留。
连续两次仍无法生成合格中文时，系统返回保守的中文回退卡片，只展示可定位原文和待
核对的方法线索，不生成未经核验的论文结论，也不会使整个批次失败。

4. 只深度分析用户选中的论文。`batch-analyze` 没有 `--topic` 参数，普通模式默认只
   生成论文档案并跳过耗时的实验矩阵：

```powershell
uv run materials-screen literature batch-analyze `
  --pdf "data\literature_pdfs\paper-1.pdf" `
  --pdf "data\literature_pdfs\paper-2.pdf"
```

命令最多接收 20 篇 PDF，逐篇隔离失败并返回稳定的 `document_id`。已有结果会复用；
只有确实需要重新尝试空矩阵时才使用 `--retry-incomplete`，不要反复重试缺少可验证数值
的定性论文。抽取规则升级后，可用 `--refresh-dossier` 显式重新生成仍处于 `pending`
的旧档案；已经批准的档案受保护，不会被覆盖。

管理员或需要结构化实验对比时，才显式追加 `--extract-matrix`。该模式会明显增加模型
调用和等待时间：

```powershell
uv run materials-screen literature batch-analyze `
  --extract-matrix `
  --pdf "data\literature_pdfs\paper-1.pdf"
```

5. 查看单篇论文档案和实验矩阵：

```powershell
uv run materials-screen literature dossier-pending <document_id> --risky-only
uv run materials-screen literature matrix-show <document_id> --summary
uv run materials-screen literature matrix-status <document_id>
```

`matrix-show --summary` 会按实验组显示指标、数值、单位和原文页码。`not_verifiable`
不算有效核验。若测量存在组别归属错误、条件缺失、单位异常或关键对比缺失，不得批准
整份矩阵。

论文档案风险分级中，逐字证据匹配且安全翻译未新增数值的候选为低风险；含有原文已
支持数值的候选为中风险；模糊引文修复、无依据摘要修复或翻译失败为高风险。数字本身
不再自动等同于高风险，证据无法定位或数字无来源的候选仍会被拒绝。

普通用户无需执行审批。使用 `user-report` 可直接从已经分析的论文生成风险过滤报告：

```powershell
uv run materials-screen literature user-report `
  --topic "多孔磷酸钙生物陶瓷骨诱导结构孔隙率" `
  --document-id <document_id_1> `
  --document-id <document_id_2>
```

该命令默认只调用一次模型，把安全证据组织为“主题概述、逐篇结论、跨论文共识、
差异、设计启示和局限性”，避免把原始候选逐条堆给普通用户。报告中的结论绑定证据
编号；程序会校验证据编号和数字是否能在被引用原文中找到。校验或模型调用失败时，
会自动退回安全的证据视图，不会输出无法追溯的综合结论。

该模式自动纳入低风险证据，将中风险证据标记为“建议核对”，隔离高风险、翻译失败和
明显误分类内容，并且不读取实验矩阵。报告 JSON 保存到
`data/literature_user_reports/`。它面向普通阅读和选文综合，不等同于正式知识库入库。
如需跳过模型调用并查看完整证据审计，可追加 `--evidence-only`：

```powershell
uv run materials-screen literature user-report `
  --topic "多孔磷酸钙生物陶瓷骨诱导结构孔隙率" `
  --document-id <document_id_1> `
  --document-id <document_id_2> `
  --evidence-only
```

6. 管理员或研究审核人员需要正式入库时，才审核论文档案。批准后，候选档案会自动
   发布到可信档案库：

```powershell
uv run materials-screen literature dossier-review <document_id> `
  --decision approved `
  --reviewer "manual-review" `
  --reason "已核对论文档案摘要与原文证据" `
  --yes
```

实验矩阵属于可选的高级结构化数据。只有全部记录均可靠时才整批批准；存在明显问题时
应拒绝，拒绝矩阵不会删除或否定已批准的论文档案：

```powershell
uv run materials-screen literature matrix-review <document_id> `
  --decision rejected `
  --reviewer "manual-review" `
  --reason "测量归属或实验条件不足，不能进入可信知识库" `
  --yes
```

7. 用户明确选择论文后生成多论文报告：

```powershell
uv run materials-screen literature result-build `
  --document-id <document_id_1> `
  --document-id <document_id_2> `
  --summary
```

报告只使用已批准内容，并自动隔离被拒绝的矩阵、安全翻译失败占位文本、明显栏目误分类
和物理上可疑的负比表面积。没有可靠实验矩阵时，已批准论文档案仍可参与多论文证据
对照，但不会生成实验数据表或可信数值知识关系。

### 使用模式与当前限制

- `batch-preview` 面向普通用户，不需要审核。
- 当前 CLI 审批流程用于开发验收、金标建设或正式知识入库；未来网页应将其实现为集中
  审核界面，而不是要求普通用户逐条输入命令。
- 单篇中文档案和多论文证据对照当前可用；通用实验矩阵对任意论文仍可能出现组别归属
  或条件缺失，不能默认自动批准。
- `user-report` 已提供带证据编号的自然语言主题综述；`result-build` 仍是面向已批准
  知识入库结果的严格结构化报告，两者用途不同。
- MinerU、图表 OCR、逐条矩阵审批和可视化网页尚未纳入当前 P4B-lite 默认流程。

实际测试命令、P4B-lite 批量流程和未来可视化网页 API/页面建议见
[LiteratureAgent README](src/materials_screening/sub_agents/literature/README.md) 与
[技术、测试及网页接入指南](docs/literature/literature_agent_user_and_web_technical_guide.md)。

## 添加新的子 Agent

统一多智能体网页、Master 路由、通用 Artifact、结果渲染注册表和跨 Agent
协作的实施基线见
[统一多智能体与可视化技术计划书](docs/multi_agent/unified_multi_agent_ui_plan.md)。
后续相关实施严格按其中的 MA-0 至 MA-5 阶段推进。

MA-5 统一多智能体页面可通过以下命令启动：

```powershell
uv run materials-screen master ui --port 8501
```

默认的“自动判断”模式由真实 `MasterAgentRunner` 在材料数据库与文献知识
子 Agent 之间选择；上传 PDF 时则依据 Artifact 类型安全交给 LiteratureAgent。
同一对话支持“深度分析第1篇”“综合这些论文”等附件指代。页面仅保存和显示
Artifact ID、文件名及处理状态，不暴露服务器内部路径。两个手动模式继续保留用于诊断。

完成论文预览或深度分析后，可输入：

```text
根据这些论文的组成查询Materials Project候选
```

Master 会从论文 dossier 生成带原文页码的 `MaterialClue`，校验元素符号后转换为
结构化数据库查询。论文组成只作为候选检索线索，不会被表述成 Materials Project
已经确认的确定化合物；联合报告分别标注文献证据和数据库查询快照。

统一页面的显式子 Agent 模式现在来自 `SubAgentUiPluginRegistry`，新增子 Agent
只需注册 `SubAgentUiSpec + handler`，无需修改数据库、文献 Agent 或主聊天分发逻辑。
页面还提供“停止”按钮、专家审计模式和统一错误恢复；专家审计只显示安全 ID、
Agent、Artifact 数量和文档数量，不显示内部文件路径。

`src/materials_screening/sub_agents/<name>/` 下创建 3 个文件：

| 文件 | 内容 |
|---|---|
| `tools.py` | 工具类，满足 `AgentTool` Protocol |
| `prompt.py` | `SYSTEM_PROMPT` 字符串（人设/技能模板） |
| `spec_factory.py` | `create_spec()` 工厂函数 → `SubAgentSpec` |

完整示例见 `sub_agents/materials_screening/`。

## 项目结构

```
src/materials_screening/
  master/                         ← Master Agent（多智能体编排）
  sub_agents/                     ← 子 Agent 容器
    materials_screening/          ← 材料筛选（5 个工具）
    literature/                   ← 文献元数据检索（OpenAlex / Semantic Scholar）
    computation/                  ← [预留] 计算任务
  agent/                          ← 单 Agent 基础设施
  agent_tools/                    ← 筛选工具实现
  planner/                        ← NL → ScreeningRequest 解析
  workflow/                       ← LangGraph 筛选工作流
  services/                       ← 确定性筛选/排序/验证/导出
  repositories/                   ← Materials Project / Mock 数据源
  config/                         ← LLM API 统一配置
```

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 请求/Schema 错误 |
| 3 | 配置错误 |
| 4 | 数据源/API 错误 |
| 5 | 验证失败 |
| 6 | 导出失败 |
| 10 | 未预期错误 |
