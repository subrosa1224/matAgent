# DA-0 数据分析契约基线与验收记录

> 对应计划：`data_analysis_agent_technical_plan.md`  
> 阶段：DA-0  
> 日期：2026-08-31  
> 状态：已验收

## 1. 本阶段范围

DA-0 只冻结数据分析 Agent 的技术路线和领域契约，不实现数据解析、统计计算、
数据清洗、Agent 工具、Master 路由或统一 UI 接入。

本阶段新增：

```text
docs/data_analysis/data_analysis_agent_technical_plan.md
docs/data_analysis/da0_contract_baseline.md
src/materials_screening/data_analysis/__init__.py
src/materials_screening/data_analysis/models.py
tests/unit/data_analysis/__init__.py
tests/unit/data_analysis/test_models.py
```

没有修改 `materials_database`、`literature`、Master、统一 UI 或现有 ArtifactRegistry
的业务实现。

## 2. 冻结的领域契约

| 契约 | 职责 |
|---|---|
| `DatasetReference` | 不暴露路径的不可变数据集版本引用 |
| `ColumnProfile` | 有界、JSON 安全的字段画像 |
| `AnalysisResult` | 关联数据集和证据的确定性分析结果 |
| `DatasetTransformOperation` | 白名单清洗操作及参数 |
| `DatasetTransformRecord` | 原始与派生数据集之间的审计记录 |

同时冻结以下字面量边界：

- 数据格式：`csv`、`json`；
- 分析类型：质量、描述、相关性、统计检验；
- 清洗操作：选择列、过滤、去重、类型转换、删除缺失和填补缺失；
- Artifact 类型：数据集、分析结果、图表和报告。

所有模型采用：

- `extra="forbid"`；
- `frozen=True`；
- 有界安全 ID；
- SHA-256 指纹格式；
- 行列数和列表规模限制；
- 非有限浮点数与非 JSON 对象拒绝；
- 派生数据集父对象与操作 ID 一致性校验。

## 3. 与现有应用契约的兼容性

现有通用应用契约无需在 DA-0 修改。定向测试确认它可以表达：

```text
owner_agent = data_analysis
artifact_type = dataset_file
result_type = analysis_answer
```

因此 DA-0 没有提前扩展 ArtifactRegistry、Renderer Registry 或 UI Registry。

## 4. 明确未实现

- CSV/JSON 文件读取；
- DatasetStore 和私有路径映射；
- 数据质量画像计算；
- 描述统计和统计检验；
- 数据清洗或文件导出；
- DataAnalysisAgent 和工具；
- Master、Artifact 与统一 UI 接入；
- 跨 Agent 适配。

这些能力严格留在 DA-1～DA-7。

## 5. 验收命令与结果

### Ruff

```powershell
.venv\Scripts\python.exe -m ruff check `
  src/materials_screening/data_analysis `
  tests/unit/data_analysis
```

结果：通过，`All checks passed!`。

### mypy

```powershell
.venv\Scripts\python.exe -m mypy `
  src/materials_screening/data_analysis
```

结果：通过，2 个源文件无类型错误。

### 契约与注册表定向回归

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/unit/data_analysis/test_models.py `
  tests/unit/master/test_application_contracts.py `
  tests/unit/master/test_sub_agent_registry.py -q
```

结果：31 项通过。Pytest 因当前工作区 `.pytest_cache` 权限产生 1 条缓存警告，
不影响测试执行和结果。

## 6. DA-0 结论

DA-0 的范围、契约、严格校验、应用层兼容性和定向回归均达到计划书门槛。
下一阶段为 DA-1：安全数据集导入、不可变存储与质量画像。
