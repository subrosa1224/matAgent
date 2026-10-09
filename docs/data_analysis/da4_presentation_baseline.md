# DA-4 可视化、报告与导出基线

> 对应计划：`data_analysis_agent_technical_plan.md`  
> 阶段：DA-4  
> 日期：2026-08-31  
> 状态：已验收

## 1. 本阶段交付

```text
src/materials_screening/data_analysis/reporting.py
tests/unit/data_analysis/test_reporting.py
```

新增 `DataAnalysisArtifact` 严格契约，DatasetStore 增加 Artifact 文件、元数据、
指纹校验和私有路径解析。本阶段没有实现 Agent、Master 或 UI 接入。

## 2. 固定图表

- histogram；
- boxplot；
- scatter；
- line；
- bar（均值与标准误差棒）；
- heatmap（Pearson/Spearman）。

绘图只接受固定字段参数，不接受任意代码、文件名、路径或 Matplotlib 参数字典。
Matplotlib 使用 Agg 后端，每次生成后在 `finally` 中关闭 Figure。散点图和折线图限制
10,000 行，热力图限制 2～20 个字段。

## 3. 报告与导出

- Markdown 报告：数据集、分析 ID、Evidence ID、结构化结果、警告、图表 Artifact
  和解释边界；
- JSON 报告：严格、无 NaN 的结构化结果；
- CSV 报告：每个分析一行，JSON 字段保持合法 CSV 引号；
- CSV/JSON 数据导出；
- CSV 文本字段防公式注入；
- 报告只能引用同一数据集的 Analysis 和 analysis_plot Artifact。

## 4. Artifact 安全

- 文件保存在私有 `artifacts/`，元数据保存在 `artifact_metadata/`；
- 公开模型不包含路径；
- Artifact ID、类型、扩展名、媒体类型、大小和 SHA-256 均有界；
- 读取时重新验证内容指纹；
- 文件或元数据写入失败时回滚本次 Artifact；
- 跨数据集分析引用和非图表引用被拒绝。

## 5. 验收命令与结果

```powershell
.venv\Scripts\python.exe -m ruff check `
  src/materials_screening/data_analysis `
  tests/unit/data_analysis

.venv\Scripts\python.exe -m mypy `
  src/materials_screening/data_analysis

.venv\Scripts\python.exe -m pytest `
  tests/unit/data_analysis `
  tests/unit/master/test_application_contracts.py `
  tests/unit/master/test_sub_agent_registry.py `
  -q --basetemp tmp/pytest-da4-run2
```

结果：

- Ruff：通过；
- mypy：8 个源文件无类型错误；
- pytest：99 项通过；
- 六类 PNG 签名、大小、资源释放以及 Markdown/JSON/CSV 可读性均通过；
- `.pytest_cache` 权限产生 1 条缓存警告，不影响隔离测试结果。

## 6. DA-4 结论

DA-4 达到计划书的固定可视化、报告、导出、Artifact 安全和前序回归门槛。
下一阶段为 DA-5：白名单工具、提示词、离线模型和 DataAnalysisAgent runner。
