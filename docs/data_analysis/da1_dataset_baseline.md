# DA-1 安全数据集与质量画像基线

> 对应计划：`data_analysis_agent_technical_plan.md`  
> 阶段：DA-1  
> 日期：2026-08-31  
> 状态：已验收

## 1. 本阶段交付

DA-1 完成了通用表格的安全导入、不可变私有存储和只读质量画像：

```text
src/materials_screening/data_analysis/parser.py
src/materials_screening/data_analysis/dataset_store.py
src/materials_screening/data_analysis/service.py
tests/unit/data_analysis/test_parser.py
tests/unit/data_analysis/test_dataset_store.py
tests/unit/data_analysis/test_service.py
```

领域契约增加 `DataQualityIssue` 和 `DatasetInspection`。本阶段没有实现描述统计、
统计检验、清洗、导出、Agent、Master 或 UI 接入。

## 2. 已冻结行为

### Parser

- 只接受 UTF-8/UTF-8 BOM CSV 和扁平对象数组 JSON；
- 默认限制 50 MB、100,000 行、500 列、5,000,000 单元格；
- 字段名不能为空、重复或超过边界；
- 拒绝空数据、坏 JSON、嵌套 JSON、非对象行和超长文本单元格；
- 返回通用 DataFrame，不映射为 `MaterialRecord`。

### DatasetStore

- 数据文件复制到私有 `datasets/`，元数据写入私有 `metadata/`；
- 对外只返回 `dataset_id`、Artifact ID、显示名、规模和指纹；
- 同内容文件通过 SHA-256 识别并复用稳定数据集；
- 元数据可在 Store 重建后恢复；
- 每次加载验证内容指纹，存储文件被篡改时拒绝分析；
- 原上传文件后续变化不会改变已登记数据集。

### Inspection

- 提供最多 100 行的有界分页预览；
- 推断 numeric、boolean、string、categorical、datetime；
- 返回非空、缺失、唯一值、样例和字段级警告；
- 识别完全重复行、常量字段、混合 Python 类型和 1.5×IQR 疑似异常值；
- NaN/Inf 不进入 JSON 结果；
- 预览、字段画像和数据集规模通过契约交叉校验。

## 3. 安全验证

- 存储目标必须是私有根目录的直接子文件；
- Artifact ID 由严格契约验证；
- 元数据不包含上传源路径；
- 原文件不被修改；
- 存储内容篡改由 SHA-256 检出；
- 非 UTF-8 和超限文件在写入私有存储前被拒绝；
- 大型数据内容不进入模型上下文。

## 4. 验收命令与结果

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
  -q --basetemp tmp/pytest-da1-run5
```

结果：

- Ruff：通过；
- mypy：5 个源文件无类型错误；
- pytest：58 项通过；
- 当前工作区 `.pytest_cache` 无写权限产生 1 条缓存警告；测试临时文件已通过
  `--basetemp` 安全隔离到项目 `tmp/`，警告不影响测试结果。

## 5. DA-1 结论

DA-1 达到计划书规定的 Parser、Store、质量画像、安全限制和前序回归门槛。
下一阶段为 DA-2：描述统计、相关性和统计检验；不会在 DA-2 实现清洗或导出。
