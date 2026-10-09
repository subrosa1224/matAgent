# DA-2 描述统计、相关性与统计检验基线

> 对应计划：`data_analysis_agent_technical_plan.md`  
> 阶段：DA-2  
> 日期：2026-08-31  
> 状态：已验收

## 1. 本阶段交付

DA-2 完成只读数值分析和分析结果持久化：

```text
src/materials_screening/data_analysis/statistics.py
tests/unit/data_analysis/test_statistics.py
```

`DatasetStore` 增加 `save_analysis` 和 `get_analysis`，`pyproject.toml` 增加独立
`analysis` 可选依赖组，`uv.lock` 已通过项目本地 uv 缓存离线更新。

本阶段没有实现数据清洗、派生数据、绘图、报告导出、Agent、Master 或 UI 接入。

## 2. 已冻结分析行为

### 描述统计

- 支持 1～50 个数值字段；
- 支持总体或最多 100 个分组，结构化结果最多 1,000 行；
- 返回有效计数、非有限/缺失计数、均值、样本标准差、最小值、最大值和分位数；
- 缺失策略固定记录为逐字段删除；
- 分位数必须唯一、升序并位于 `[0, 1]`。

### 相关性

- 支持 Pearson 和 Spearman；
- 最多 40 个数值字段；
- 返回上三角字段对、成对有效样本数和相关系数；
- 缺失策略固定为 pairwise complete；
- 常量字段、非有限结果和样本不足返回 `None` 并产生有界警告。

### 统计检验

已实现：

- 独立样本 Student t；
- 独立样本 Welch t；
- 配对 t；
- Mann–Whitney U；
- 单因素 ANOVA；
- Kruskal–Wallis。

所有结果包含实际方法、参数、组标签、每组样本量、统计量、p 值、显著性水平、
效应量、多重检验校正状态和适用的置信区间/自由度。参数检验返回 Shapiro
正态性和 Levene 方差检查；Welch 不把方差不齐误报为方法失败；非参数检验明确
记录独立性由用户研究设计保证。

效应量：

- Student/Welch t：Cohen's d；
- 配对 t：Cohen's dz；
- Mann–Whitney：rank-biserial correlation；
- ANOVA：eta squared；
- Kruskal–Wallis：epsilon squared。

服务不会根据结果静默替换用户指定检验。

## 3. 结果安全与持久化

- 每个结果关联 `dataset_id`、`analysis_id` 和 `evidence_id`；
- 分析 JSON 只写入私有 `analyses/`；
- 分析 ID 路径穿越被拒绝；
- 已存在 ID 只接受完全相同结果；
- NaN/Inf 不进入结构化结果；
- 警告数量有界；
- 分析结果经过 DA-0 的严格 JSON 契约验证后才持久化。

## 4. 依赖

```toml
analysis = [
  "scipy>=1.15,<2",
  "matplotlib>=3.10,<4",
]
```

DA-2 只使用 SciPy；Matplotlib 留给计划内 DA-4，不在本阶段提前绘图。

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
  -q --basetemp tmp/pytest-da2-run3
```

结果：

- Ruff：通过；
- mypy：6 个源文件无类型错误；
- pytest：70 项通过；
- 固定小样本与参考统计量、p 值和效应量对照通过；
- `.pytest_cache` 权限产生 1 条缓存警告，测试临时目录已安全隔离，不影响结果。

## 6. DA-2 结论

DA-2 达到计划书的只读统计、方法边界、结果可追溯性、依赖和前序回归门槛。
下一阶段为 DA-3：不可变清洗与派生数据；不会在 DA-3 提前实现绘图或 Agent。
