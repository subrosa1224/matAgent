# LiteratureAgent P4B-lite 批量逐篇分析流程

> 状态：完成  
> 日期：2026-08-21  
> 设计原则：先逐篇形成独立证据档案，再由用户选择论文；禁止上传后直接混合总结。

## 1. 固定工作流

1. 用户用 P4A 检索候选论文并合法获取 PDF。
2. 用户把一批 PDF 放入 `LITERATURE_INGEST_ROOTS` 配置的目录。
3. `batch-preview` 每篇只调用一次轻量模型，给出相关性与深度分析建议，不要求审核。
4. 用户选择真正需要的论文，再用 `batch-analyze` 生成完整档案；实验矩阵由管理员
   使用 `--extract-matrix` 按需生成。
5. 系统显示深度分析批次汇总；单篇失败只标记该篇，不中止其他论文。
6. 用户只处理选中文献的关键异常，并明确选择用于 `result-build` 的论文。

快速阅读命令：

```powershell
uv run materials-screen literature batch-preview `
  --topic "多孔生物陶瓷孔结构与成骨性能" `
  --directory data\literature_pdfs
```

预览不是已批准科学事实，不进入知识关系；它只用于筛选论文。缓存按 PDF 内容与主题共同
区分，换主题会重新生成预览，同一主题重跑则直接复用。

快速预览采用宽容降级：模型引文无法逐字定位时，不会丢弃整篇预览，而是展示对应原始
chunk 并标记 `fallback_chunk`；概述中不受代表引文支持的数字会替换为“数值待深度
分析”。这些内容只能帮助选文，精确数值仍必须进入深度分析和证据审核。

## 2. 批量分析

分析目录中的 PDF：

```powershell
uv run materials-screen literature batch-analyze `
  --directory data\literature_pdfs
```

也可以重复传入 `--pdf`，每批最多 20 篇。命令以 PDF 内容哈希生成稳定文档 ID；已有档案
会直接复用，因而支持断点续跑，不会仅因重新执行批次命令而重复调用模型。普通模式
默认跳过实验矩阵；显式使用 `--extract-matrix` 时才生成或复用矩阵。

批次会先识别所有尚未索引的 PDF，只加载一次 bge-m3 并统一入库；不会再为每篇 PDF
重复加载向量模型。已经索引的批次会直接显示“跳过向量模型加载”。批次输出默认只显示
证据门警告数量，逐条警告仅在单独运行 `matrix-extract --show-warnings` 时展示。

若某篇已有实验组但测量数为 0，可显式只重试不完整矩阵：

```powershell
uv run materials-screen literature batch-analyze `
  --pdf "data\literature_pdfs\paper.pdf" `
  --retry-incomplete
```

该选项会调用抽取模型，应由用户主动触发；默认重跑批次不会额外消耗这部分额度。

批次结果保存在 `data/literature_batches/<batch_id>.json`，可重新查看：

```powershell
uv run materials-screen literature batch-show <batch_id>
```

汇总为每篇论文显示档案条数、完整度、实验组数、测量数、待审核数、状态和错误。结果始终
按 `document_id` 隔离，不跨论文绑定实验条件或证据。

## 3. 查看、审核与用户选择

```powershell
uv run materials-screen literature dossier-pending <document_id>
uv run materials-screen literature matrix-show <document_id> --summary
```

审核完成后，用户显式选择论文生成报告：

```powershell
uv run materials-screen literature result-build `
  --document-id <document_id_1> `
  --document-id <document_id_2> `
  --summary
```

未被用户指定的论文不会进入报告。正式报告只读取已批准档案和实验数据；待审核内容不能
成为可信知识关系。

## 4. 验收记录

- 批次持久化、稳定 ID、状态计数和单篇自动分析相关离线测试共 18 项通过。
- Ruff 与新增批次模块定向 mypy 通过。
- 使用三篇现有金标 PDF 做无模型复用验收，均识别为完成：3 份档案、32 个实验组、
  68 条测量，待审核测量为 0。
- 验收批次 ID：`lit-batch-09b1dca5091bf0e2d54354d1`。

P4B-lite 不负责自动下载付费全文，也不自动决定哪些论文应该进行跨论文比较。
