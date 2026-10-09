# LiteratureAgent P0 现状基线

> 基线日期：2026-08-20  
> 对应阶段：P0 冻结现状并建立可信基线  
> 阶段状态：已完成，待用户确认进入P1  
> 依据：`literature_agent_execution_baseline.md`  
> 范围：只记录已存在能力、数据和缺陷；不宣称通用自动化已经完成。

## 1. 环境基线

`materials-screen literature doctor` 的实测结果：

| 项目 | 状态 |
| --- | --- |
| PyMuPDF | 已安装 |
| psycopg | 已安装 |
| sentence-transformers | 已安装 |
| FlagEmbedding | 已安装 |
| PDF 摄取目录 | `data/literature_pdfs` 可用 |
| PostgreSQL/pgvector | 可连接，pgvector 0.8.6 |
| 元数据检索依赖 | 就绪 |
| PDF RAG 依赖 | 就绪 |

当前解析器仅为 PyMuPDF。MinerU 未安装、未接入，按照执行基线延后评估。

## 2. 命令清单与稳定性

| 命令 | 状态 | 用途/限制 |
| --- | --- | --- |
| `doctor` | 可用 | 检查本地依赖、摄取目录和 pgvector |
| `migrate` | 可用 | 创建或更新 LiteratureAgent 数据库对象 |
| `document-metadata` | 可用 | 为已入库 PDF 绑定经核实的题名、DOI 和年份 |
| `facts-list` | 可用（旧事实模型） | 查看旧式实验事实 |
| `facts-review` | 可用（旧事实模型） | 审核旧式实验事实并记录审计事件 |
| `edges-build` | 可用 | 仅从已批准证据生成知识关系 |
| `edges-list` | 可用 | 查看已生成知识关系 |
| `dossier-show` | 可用 | 查看单篇论文档案 |
| `dossier-review` | 可用 | 审核档案并保存审计事件 |
| `matrix-status` | 可用 | 查看新实验矩阵记录数量 |
| `matrix-show` | 可用 | 查看实验组、测量、比较和摘要核验 |
| `matrix-review` | 可用 | 批量审核一篇论文的矩阵记录 |
| `result-build` | 部分可用 | 能组装已批准数据；主题导向综合尚未完成 |
| `process` | 实验性 | 通用自动档案抽取未通过验收，不能作为正式入口 |

当前没有面向普通用户的统一 `literature search` 命令。OpenAlex 和 Semantic Scholar
是智能体工具层的底层能力，并不等于主题检索产品流程已经闭合。

## 3. 代码模块清单

| 模块 | 当前职责 | 状态 |
| --- | --- | --- |
| `providers.py` | OpenAlex/Semantic Scholar API 适配 | 基础实现 |
| `service.py` | 单 Provider 查询、去重、排序和快照 | 部分实现 |
| `store.py` | 本地查询快照 | 已实现；当前没有真实查询快照 |
| `rag.py` | PyMuPDF、bge-m3、重排、摄取和检索 | 基础设施已实现 |
| `pgvector_store.py` | 文档、chunk、事实、矩阵和审核持久化 | 已实现 |
| `extraction.py` | 旧事实候选的证据校验和暂存 | 已实现 |
| `dossier.py` | 单篇档案模型、存储和审核 | 已实现 |
| `matrix.py` | 新实验矩阵证据校验 | 已实现 |
| `comparison.py` | 从测量值确定性生成比较 | 已实现 |
| `claim_assessment.py` | 摘要主张与正文证据关联 | 基础实现 |
| `integration.py` | 从已批准数据组装 LiteratureResult | 部分实现 |
| `automation.py` | LLM 自动档案抽取原型 | 实验性、不稳定 |
| `tools.py` / `spec_factory.py` | 工具注册和子智能体构造 | 基础实现 |

## 4. 数据库对象

当前 migration 定义 11 张表：

1. `literature_documents`
2. `literature_chunks`
3. `literature_experimental_facts`
4. `literature_extraction_reviews`
5. `literature_knowledge_edges`
6. `literature_experimental_groups`
7. `literature_measurements`
8. `literature_comparisons`
9. `literature_paper_claims`
10. `literature_claim_evidence_links`
11. `literature_matrix_reviews`

`literature_chunks.embedding` 使用 `vector(1024)`，已定义 HNSW cosine 索引。数据库
连接由 `LITERATURE_DATABASE_URL` 提供，诊断和日志不得输出密码。

## 5. 三篇金标的准确状态

| 文档 | 档案 | 新实验矩阵 | 旧事实 | 用途 |
| --- | ---: | ---: | ---: | --- |
| Nb 掺杂 TiO2 | 12 项，已批准 | 4 组、24 测量、18 比较、1 主张，已批准 | 0 | P2/P3 金标 |
| 混合阳离子钙钛矿 | 16 项，已批准 | 3 组、7 测量、1 比较、1 主张，已批准 | 0 | P2/P3 金标 |
| 2025 生物陶瓷 | 14 项，已批准 | 25 组、37 测量、5 比较、1 主张，已批准 | 3 条，已批准 | P2/P3 金标 |

生物陶瓷的 3 条记录是旧式事实，分别覆盖 70% 孔隙率、比表面积范围和渗透率，不能
再描述为“3 条新矩阵测量”。经用户确认的P0范围修正已经生成并批准25组、37测量、
5比较的新矩阵；金标夹具已记录两套模型的准确数量，后续评测以新矩阵为主。

当前 `data/literature_pdfs` 中有 4 个 PDF；其中额外的电池论文不是已批准金标。当前
`data/literature_queries` 中没有真实查询快照，因此不能据此声称主题检索已经完成验证。

## 6. 自动化质量基线

2026-08-20 在项目虚拟环境中执行：

| 检查 | 结果 |
| --- | --- |
| 11 个 `test_literature_*.py` 文件 | 31 passed |
| LiteratureAgent 源码及对应测试 Ruff | 全部通过 |
| LiteratureAgent 包 + `cli.py` + 工具注册定向 mypy | 通过，0 错误 |
| 全项目 `mypy src` | 26 个既有错误，分布在 11 个非 LiteratureAgent 文件 |

首次 pytest 在系统临时目录产生 10 个 `PermissionError`；指定项目内 `--basetemp` 后
31 项全部通过，确认这是执行环境权限问题，不是测试断言失败。后续本机基线命令必须使用
项目内临时目录，避免把环境错误误报为代码缺陷。

全项目 mypy 的 26 个错误属于材料数据库、离群检测、Master、UI 等既有模块；本阶段不
跨范围修复。P0 的约束是 LiteratureAgent 后续不得增加新的 mypy 错误。

## 7. 已知缺陷与边界

| 编号 | 缺陷/边界 | 严重度 | 计划处理阶段 |
| --- | --- | --- | --- |
| LIT-001 | 缺少结构化查询扩写器 | 阻塞主题入口 | P1 |
| LIT-002 | OpenAlex/S2 尚未合并成统一结果集 | 阻塞主题入口 | P1 |
| LIT-003 | 没有正式的自然语言检索命令和可复现统一 query_id | 阻塞主题入口 | P1 |
| LIT-004 | `automation.py` 在真实模型调用中出现结构化输出失败 | 阻塞 PDF 自动分析 | P2 |
| LIT-005 | 尚无通用实验矩阵自动抽取器 | 阻塞实验对比自动化 | P3 |
| LIT-006 | 生物陶瓷主文图表缺少24组完整精确数值，当前金标不做图像估读 | 不阻塞首版 | 如P3评测需要，再引入补充材料 |
| LIT-007 | 综合报告主要拼接已批准内容，缺少研究问题导向的跨论文综合 | 阻塞最终产品 | P4 |
| LIT-008 | PyMuPDF 对复杂表格、双栏、公式和扫描件支持有限 | 非首版阻塞 | P5 后评估 MinerU |
| LIT-009 | 没有 10 主题正式金标和评测结果 | 阻塞发布 | P5 |
| ENV-001 | 默认 pytest 临时目录在当前 Windows 环境可能无权限 | 可规避 | 固定项目内 basetemp |
| REPO-001 | 全项目 mypy 当前有 26 个既有错误 | 非本 Agent 阻塞 | 单独治理 |

## 8. P0 结论

P0 只证明以下事实：

- LiteratureAgent 的本地依赖、pgvector、核心数据模型、审核机制和定向测试基线可用。
- 两篇论文具备已批准的新实验矩阵，第三篇只有已批准档案和旧事实。
- 自动主题检索、通用 PDF 自动分析、通用实验矩阵抽取和最终端到端流程尚未完成。
- `process` 已明确标注为实验性。

本基线通过后，下一阶段只能进入 P1：统一文献检索。不得跳到 P2/P3，也不得继续用
手工论文拆分代替通用能力。
