# DA-3 不可变清洗与派生数据基线

> 对应计划：`data_analysis_agent_technical_plan.md`  
> 阶段：DA-3  
> 日期：2026-08-31  
> 状态：已验收

## 1. 本阶段交付

DA-3 新增结构化、可审计的数据清洗服务：

```text
src/materials_screening/data_analysis/transform.py
tests/unit/data_analysis/test_transform.py
```

`DatasetStore` 增加派生数据集、元数据和 TransformRecord 的原子保存与读取。
本阶段没有实现绘图、报告导出、Agent、Master 或 UI 接入。

## 2. 清洗白名单

- `select_columns`：显式选择字段；
- `filter_rows`：`eq/ne/gt/ge/lt/le/in/not_in` 固定操作符；
- `drop_duplicates`：可选字段集合与 first/last；
- `convert_type`：numeric/string/boolean/datetime；
- `drop_missing`：可选字段集合与 any/all；
- `fill_missing`：常量、总体均值/中位数、分组均值/中位数。

所有操作拒绝额外参数、未知字段、任意表达式、嵌套值和非有限数值。
操作按声明顺序执行，后续操作只能看到前序操作产生的 Schema。

## 3. 不可变性与原子性

- 每次成功清洗生成新的随机 `dataset_id`；
- 新 DatasetReference 同时记录 `parent_dataset_id` 和 `operation_id`；
- 原始文件、原始内容哈希和元数据保持不变；
- 派生 CSV、DatasetReference 和 TransformRecord 作为一个提交单元；
- 预校验、序列化或提交失败时回滚本次新文件；
- 重复 operation ID 被拒绝，已有成功结果不受影响；
- Store 重启后仍可恢复派生数据集和操作记录；
- 清洗完成后自动重新运行 DA-1 质量检查。

## 4. CSV 安全

派生 CSV 中以 `= + - @` 开头的文本单元格会加前导单引号，并在 TransformRecord
记录转义数量。数值负号不受影响。该行为防止派生 CSV 被电子表格应用打开时触发
公式执行。

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
  -q --basetemp tmp/pytest-da3-run2
```

结果：

- Ruff：通过；
- mypy：7 个源文件无类型错误；
- pytest：84 项通过；
- `.pytest_cache` 权限产生 1 条缓存警告，隔离测试目录正常工作。

## 6. DA-3 结论

DA-3 达到计划书规定的白名单、预校验、不可变派生链、原子写入、CSV 安全、
质量复查和前序回归门槛。下一阶段为 DA-4：可视化、报告和导出。
