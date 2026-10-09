# 无机半导体材料筛选系统——第一阶段技术细节文档

> 项目名称：`materials-screening-core`  
> 阶段定义：**实施阶段 1——纯规则单体版本**  
> 文档版本：v1.0  
> 核对日期：2026-08-04  
> 目标用户：使用 Codex 或 Claude Code 辅助开发的计算机专业学生

---

## 0. 本阶段范围说明

本阶段只实现一个**确定性、可测试、可复现的材料筛选内核**。

本阶段包含：

1. 使用结构化 JSON 描述筛选条件；
2. 使用 Materials Project 官方 `mp-api` 查询材料；
3. 使用 Python 硬规则二次过滤；
4. 对候选材料进行可解释排序；
5. 验证结果是否满足约束并保留数据来源；
6. 导出 JSON、CSV 和 CIF；
7. 提供命令行接口；
8. 提供完整单元测试、Mock 集成测试和一个手动真实 API 冒烟测试。

本阶段明确不包含：

- 自然语言解析；
- 大语言模型；
- 多智能体；
- LangGraph；
- RAG；
- FastAPI；
- 数据库服务；
- Web 前端；
- JARVIS、OQMD 等多数据源；
- 属性预测模型；
- 自动提交 DFT 任务。

虽然最终项目目标是多智能体系统，但第一阶段先构建可靠的“工具层和规则层”。后续智能体只能调用本阶段提供的接口，不能替代这些确定性规则。

---

# 1. 第一阶段最终成果

完成后，用户应能执行：

```bash
uv run materials-screen screen \
  --request examples/semiconductor_request.json \
  --output data/exports
```

输入文件：

```json
{
  "required_elements": [],
  "excluded_elements": ["Pb", "Cd", "Hg"],
  "chemsys": null,
  "formula": null,
  "band_gap_ev": {
    "min": 1.2,
    "max": 2.0
  },
  "energy_above_hull_ev_atom": {
    "min": 0.0,
    "max": 0.05
  },
  "density_g_cm3": null,
  "crystal_system": null,
  "spacegroup_numbers": [],
  "is_metal": false,
  "is_stable": null,
  "theoretical": null,
  "target_band_gap_ev": 1.6,
  "limit": 10
}
```

输出目录示例：

```text
data/exports/run_20260804_172200/
├── request.json
├── result.json
├── candidates.csv
├── report.md
├── provenance.json
└── cif/
    ├── mp-xxxx.cif
    └── mp-yyyy.cif
```

终端应显示：

```text
Task completed
Retrieved: 186
Passed hard filters: 23
Returned: 10
Validation: PASSED
Output: data/exports/run_20260804_172200
```

---

# 2. 技术栈

## 2.1 运行环境

推荐：

```text
Python 3.11
Windows 11 / Ubuntu 22.04+
uv
```

Python 3.11 的原因：

- 科学计算生态兼容性较稳；
- `mp-api`、`pymatgen`、Pydantic v2 支持良好；
- 避免过早使用较新的 Python 版本造成二进制依赖兼容问题。

## 2.2 核心依赖

建议由代码智能体在实施时查询当前兼容版本并生成 `uv.lock`，不要机械复制过时锁版本。

```toml
[project]
name = "materials-screening-core"
version = "0.1.0"
description = "Deterministic inorganic semiconductor screening core"
requires-python = ">=3.11,<3.13"
dependencies = [
    "mp-api>=0.46,<0.47",
    "pymatgen>=2026.5",
    "pydantic>=2.13,<3",
    "pydantic-settings>=2,<3",
    "pandas>=2.2,<3",
    "typer>=0.16,<1",
    "rich>=14,<15",
    "tenacity>=9,<10",
    "orjson>=3.10,<4",
]

[dependency-groups]
dev = [
    "pytest>=8,<9",
    "pytest-cov>=6,<7",
    "ruff>=0.12,<1",
    "mypy>=1.17,<2",
]
```

注意：

1. 上述版本范围是 2026-08-04 的建议基线；
2. 以 `uv lock` 实际解析成功结果为准；
3. 若 `mp-api` 与 `pymatgen` 依赖冲突，优先让 `uv` 求解兼容组合；
4. 不直接安装 GitHub 主分支；
5. 不允许代码智能体为了消除冲突而删除核心类型检查或测试依赖。

---

# 3. 项目目录

```text
materials-screening-core/
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
│
├── examples/
│   ├── semiconductor_request.json
│   ├── li_fe_o_request.json
│   └── invalid_request.json
│
├── src/
│   └── materials_screening/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── config.py
│       ├── errors.py
│       ├── models.py
│       ├── chemistry.py
│       ├── fingerprints.py
│       │
│       ├── repositories/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── mock.py
│       │   └── materials_project.py
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   ├── filter_service.py
│       │   ├── ranking_service.py
│       │   ├── validation_service.py
│       │   ├── export_service.py
│       │   └── screening_service.py
│       │
│       └── formatting/
│           ├── __init__.py
│           └── markdown_report.py
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── mp_documents.json
│   │   ├── requests.json
│   │   └── expected_results.json
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_chemistry.py
│   │   ├── test_fingerprints.py
│   │   ├── test_mp_mapping.py
│   │   ├── test_filter_service.py
│   │   ├── test_ranking_service.py
│   │   ├── test_validation_service.py
│   │   └── test_export_service.py
│   ├── integration/
│   │   ├── test_screening_service_mock.py
│   │   └── test_cli.py
│   └── manual/
│       └── test_real_mp_api.py
│
├── data/
│   ├── cache/
│   └── exports/
│
└── scripts/
    └── inspect_mp_fields.py
```

---

# 4. 模块边界

主调用链：

```text
CLI
  ↓
ScreeningService
  ↓
MaterialsRepository
  ↓
MaterialRecord[]
  ↓
FilterService
  ↓
RankingService
  ↓
ValidationService
  ↓
ExportService
```

设计规则：

- CLI 不包含业务逻辑；
- Repository 不排序；
- FilterService 不访问网络；
- RankingService 不访问网络；
- ValidationService 不修改候选；
- ExportService 不重新计算属性；
- ScreeningService 只负责协调上述服务；
- 所有服务输入输出使用 Pydantic 模型；
- 真实 API 和 Mock API 实现相同 Protocol。

---

# 5. 领域数据模型

## 5.1 数值范围

```python
from pydantic import BaseModel, ConfigDict, model_validator


class FloatRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_order(self) -> "FloatRange":
        if self.min is None and self.max is None:
            raise ValueError("At least one bound must be provided")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot be greater than max")
        return self
```

带隙额外要求：

```text
min >= 0
max >= 0
```

能量高于凸包第一版要求：

```text
min >= 0
max >= 0
```

## 5.2 筛选请求

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScreeningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    required_elements: tuple[str, ...] = ()
    excluded_elements: tuple[str, ...] = ()

    chemsys: str | None = None
    formula: str | None = None

    band_gap_ev: FloatRange | None = None
    energy_above_hull_ev_atom: FloatRange | None = None
    density_g_cm3: FloatRange | None = None

    crystal_system: str | None = None
    spacegroup_numbers: tuple[int, ...] = ()

    is_metal: bool | None = False
    is_stable: bool | None = None
    theoretical: bool | None = None

    target_band_gap_ev: float | None = None
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("required_elements", "excluded_elements")
    @classmethod
    def normalize_elements(cls, values: tuple[str, ...]) -> tuple[str, ...]: ...
```

业务校验：

1. 同一元素不能同时出现在 required 和 excluded；
2. 元素符号使用 `pymatgen.core.Element` 校验；
3. 元素顺序标准化并去重；
4. `chemsys` 应标准化为例如 `Li-Fe-O`；
5. 空字符串转为 `None`；
6. `target_band_gap_ev` 不得为负；
7. 若目标带隙超出明确带隙区间，应报错；
8. 空间群编号范围为 1–230；
9. 晶系只接受规范枚举。

推荐晶系枚举：

```python
class CrystalSystem(str, Enum):
    TRICLINIC = "Triclinic"
    MONOCLINIC = "Monoclinic"
    ORTHORHOMBIC = "Orthorhombic"
    TETRAGONAL = "Tetragonal"
    TRIGONAL = "Trigonal"
    HEXAGONAL = "Hexagonal"
    CUBIC = "Cubic"
```

## 5.3 来源记录

```python
from datetime import datetime
from enum import Enum


class PropertyValueType(str, Enum):
    DFT_CALCULATED = "dft_calculated"
    EXPERIMENTAL = "experimental"
    ML_PREDICTED = "ml_predicted"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class PropertyProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    property_name: str
    source: str
    source_material_id: str
    value_type: PropertyValueType
    database_version: str | None
    retrieved_at: datetime
    method: str | None = None
```

第一阶段所有 Materials Project 数值默认标为：

```text
dft_calculated
```

不要把 `formula_pretty`、材料 ID 等身份字段归类为计算属性。

## 5.4 材料记录

```python
class SymmetryInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    crystal_system: CrystalSystem | None = None
    symbol: str | None = None
    number: int | None = Field(default=None, ge=1, le=230)


class MaterialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    material_id: str
    formula_pretty: str
    elements: tuple[str, ...]
    chemsys: str | None = None

    band_gap_ev: float | None = Field(default=None, ge=0)
    energy_above_hull_ev_atom: float | None = Field(default=None, ge=0)
    formation_energy_ev_atom: float | None = None
    density_g_cm3: float | None = Field(default=None, gt=0)

    is_metal: bool | None = None
    is_gap_direct: bool | None = None
    is_stable: bool | None = None
    theoretical: bool | None = None
    deprecated: bool | None = None

    symmetry: SymmetryInfo | None = None

    structure_dict: dict | None = None
    structure_hash: str | None = None

    provenance: tuple[PropertyProvenance, ...] = ()
```

不要在该模型中保存任意不受控 `raw_fields`，否则后续代码容易绕过标准字段。若确实需要调试原始字段，只在 Repository 内部日志中以安全摘要方式记录。

## 5.5 过滤轨迹

```python
class Rejection(BaseModel):
    material_id: str
    reasons: tuple[str, ...]


class FilterStep(BaseModel):
    name: str
    before_count: int
    after_count: int
    rejection_count: int
    reason_counts: dict[str, int]


class FilterTrace(BaseModel):
    steps: tuple[FilterStep, ...]
    rejections: tuple[Rejection, ...]
```

## 5.6 排名结果

```python
class ScoreBreakdown(BaseModel):
    stability: float
    band_gap_match: float
    completeness: float
    direct_gap: float

    weighted_stability: float
    weighted_band_gap_match: float
    weighted_completeness: float
    weighted_direct_gap: float


class RankedMaterial(BaseModel):
    record: MaterialRecord
    rank: int = Field(ge=1)
    total_score: float = Field(ge=0, le=1)
    score_breakdown: ScoreBreakdown
```

## 5.7 验证报告

```python
class ValidationReport(BaseModel):
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checked_material_ids: tuple[str, ...] = ()
```

## 5.8 最终结果

```python
class RunMetadata(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime
    source: str
    database_version: str | None
    mp_api_version: str | None
    pymatgen_version: str | None
    application_version: str
    query_fingerprint: str


class ScreeningResult(BaseModel):
    request: ScreeningRequest
    metadata: RunMetadata
    retrieved_count: int
    passed_filter_count: int
    ranked_materials: tuple[RankedMaterial, ...]
    filter_trace: FilterTrace
    validation: ValidationReport
```

---

# 6. 错误体系

定义项目异常：

```python
class MaterialsScreeningError(Exception):
    """Base project exception."""


class ConfigurationError(MaterialsScreeningError):
    pass


class InvalidRequestError(MaterialsScreeningError):
    pass


class RepositoryError(MaterialsScreeningError):
    pass


class RepositoryAuthenticationError(RepositoryError):
    pass


class RepositoryRateLimitError(RepositoryError):
    pass


class RepositoryTimeoutError(RepositoryError):
    pass


class RepositoryMappingError(RepositoryError):
    pass


class ValidationFailedError(MaterialsScreeningError):
    pass


class ExportError(MaterialsScreeningError):
    pass
```

错误处理规则：

- 不使用裸 `except Exception: pass`；
- 不静默返回空列表；
- 认证失败不能重试；
- 请求参数错误不能重试；
- 超时和部分 5xx 可以有限重试；
- 每个错误对 CLI 映射到稳定退出码。

建议退出码：

```text
0  成功
2  请求文件或 Schema 错误
3  配置错误
4  Repository/API 错误
5  结果验证失败
6  导出失败
10 未预期错误
```

---

# 7. Materials Project Repository

## 7.1 官方接口基线

使用：

```python
from mp_api.client import MPRester
```

主要查询：

```python
mpr.materials.summary.search(...)
```

当前官方 Summary API 支持的本项目相关查询参数包括：

```text
band_gap
chemsys
crystal_system
density
deprecated
elements
energy_above_hull
exclude_elements
formula
is_gap_direct
is_metal
is_stable
spacegroup_number
theoretical
chunk_size
all_fields
fields
```

实现前代码智能体必须执行：

```bash
uv run python scripts/inspect_mp_fields.py
```

脚本输出当前客户端的：

- `SummaryRester.search` 签名；
- `mpr.materials.summary.available_fields`；
- `mp-api` 版本；
- `pymatgen` 版本；
- Materials Project 数据库版本。

若字段变化，应更新映射测试并记录到 `README.md`，不得猜测。

## 7.2 API Key

`.env.example`：

```dotenv
MP_API_KEY=
```

配置：

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mp_api_key: SecretStr | None = None
    mp_timeout_seconds: int = 30
    mp_max_attempts: int = 3
    export_root: Path = Path("data/exports")
```

规则：

- `.env` 必须加入 `.gitignore`；
- 日志不得打印 `SecretStr.get_secret_value()`；
- 单元测试不得要求真实 Key；
- 手动真实 API 测试在无 Key 时 skip；
- 可兼容 `PMG_MAPI_KEY`，但项目内部统一优先读取 `MP_API_KEY`。

## 7.3 Repository Protocol

```python
from typing import Protocol


class MaterialsRepository(Protocol):
    def search(self, request: ScreeningRequest) -> RetrievalResult: ...

    def healthcheck(self) -> bool: ...
```

返回模型：

```python
class RetrievalResult(BaseModel):
    source: str
    database_version: str | None
    retrieved_at: datetime
    records: tuple[MaterialRecord, ...]
    warnings: tuple[str, ...] = ()
```

## 7.4 查询参数构建

```python
def build_mp_query(request: ScreeningRequest) -> dict[str, object]:
    query: dict[str, object] = {
        "deprecated": False,
        "all_fields": False,
        "fields": [
            "material_id",
            "formula_pretty",
            "elements",
            "chemsys",
            "band_gap",
            "energy_above_hull",
            "formation_energy_per_atom",
            "density",
            "is_metal",
            "is_gap_direct",
            "is_stable",
            "theoretical",
            "deprecated",
            "symmetry",
            "structure",
        ],
        "chunk_size": 500,
    }

    if request.band_gap_ev is not None:
        query["band_gap"] = range_to_mp_tuple(
            request.band_gap_ev,
            default_min=0.0,
            default_max=100.0,
        )

    if request.energy_above_hull_ev_atom is not None:
        query["energy_above_hull"] = range_to_mp_tuple(
            request.energy_above_hull_ev_atom,
            default_min=0.0,
            default_max=100.0,
        )

    if request.density_g_cm3 is not None:
        query["density"] = range_to_mp_tuple(
            request.density_g_cm3,
            default_min=0.0,
            default_max=1000.0,
        )

    if request.required_elements:
        query["elements"] = list(request.required_elements)

    if request.excluded_elements:
        query["exclude_elements"] = list(request.excluded_elements)

    if request.chemsys:
        query["chemsys"] = request.chemsys

    if request.formula:
        query["formula"] = request.formula

    if request.crystal_system:
        query["crystal_system"] = request.crystal_system

    if request.spacegroup_numbers:
        query["spacegroup_number"] = list(request.spacegroup_numbers)

    if request.is_metal is not None:
        query["is_metal"] = request.is_metal

    if request.is_stable is not None:
        query["is_stable"] = request.is_stable

    if request.theoretical is not None:
        query["theoretical"] = request.theoretical

    return query
```

说明：

- API 端筛选用于降低下载量；
- 本地 FilterService 必须再次执行全部硬约束；
- 不信任远端返回一定满足全部条件；
- 不把 `limit` 直接当作“只获取前 N 条”，因为应先完成本地排序；
- 初版可以设置最大检索数量保护，例如 10,000；
- 若查询过宽可能超过保护值，明确报错并要求用户增加条件。

## 7.5 查询实现

```python
class MaterialsProjectRepository:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 30,
        max_attempts: int = 3,
    ) -> None: ...

    def search(self, request: ScreeningRequest) -> RetrievalResult:
        query = build_mp_query(request)

        try:
            with MPRester(
                self._api_key,
                timeout=self._timeout_seconds,
                mute_progress_bars=True,
            ) as mpr:
                database_version = mpr.get_database_version()
                documents = self._search_with_retry(mpr, query)
        except ...:
            ...

        records = tuple(
            map_summary_document(document, database_version) for document in documents
        )

        return RetrievalResult(
            source="materials_project",
            database_version=database_version,
            retrieved_at=datetime.now(timezone.utc),
            records=records,
        )
```

不要在 Repository 中截断为 `request.limit`。

## 7.6 字段映射

映射器必须同时支持：

- SummaryDoc 对象；
- fixture 中的普通 dict。

辅助函数：

```python
def read_field(obj: object, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
```

材料 ID：

```python
material_id = str(read_field(doc, "material_id"))
```

元素：

```python
elements = tuple(sorted(str(element) for element in read_field(doc, "elements", [])))
```

对结构字段：

```python
structure = read_field(doc, "structure")
if structure is None:
    structure_dict = None
elif isinstance(structure, dict):
    structure_dict = structure
elif hasattr(structure, "as_dict"):
    structure_dict = structure.as_dict()
else:
    raise RepositoryMappingError(...)
```

对 symmetry 同样兼容 dict 和对象。

## 7.7 数据来源

至少为下列字段建立 provenance：

```text
band_gap_ev
energy_above_hull_ev_atom
formation_energy_ev_atom
density_g_cm3
is_metal
is_gap_direct
is_stable
theoretical
symmetry
structure
```

每个 provenance 使用：

```text
source = materials_project
source_material_id = mp-...
value_type = dft_calculated
database_version = 当前版本
retrieved_at = 本次查询时间
```

对于 `is_metal` 和结构等非单纯数值字段，仍可记录来源，但 `value_type` 的命名应在 README 说明其表示数据库计算/派生结果类别。

---

# 8. Mock Repository

Mock Repository 是本阶段测试核心。

```python
class MockMaterialsRepository:
    def __init__(self, records: tuple[MaterialRecord, ...]) -> None:
        self._records = records

    def search(self, request: ScreeningRequest) -> RetrievalResult:
        return RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=FIXED_TEST_TIME,
            records=self._records,
        )

    def healthcheck(self) -> bool:
        return True
```

测试中 Repository 可以返回“故意不满足请求”的记录，以证明本地过滤器有效。

fixture 至少包含：

1. 符合全部条件的材料；
2. 含 Pb；
3. 含 Cd；
4. 带隙过低；
5. 带隙过高；
6. hull 过高；
7. 金属；
8. 缺少带隙；
9. 缺少 hull；
10. deprecated；
11. 相同属性不同 ID；
12. 完全相同 ID 重复；
13. 无 structure；
14. 非直接带隙；
15. 理论材料。

---

# 9. 硬约束过滤

## 9.1 原则

缺失一个用户明确要求的属性时，默认拒绝，而不是猜测通过。

例如用户要求：

```text
band_gap 1.2–2.0 eV
```

材料 `band_gap=None` 必须拒绝，理由为：

```text
missing_band_gap
```

## 9.2 过滤顺序

固定顺序：

1. deprecated；
2. 重复 ID；
3. 排除元素；
4. 必需元素；
5. chemsys；
6. formula；
7. is_metal；
8. is_stable；
9. theoretical；
10. crystal_system；
11. spacegroup；
12. band_gap；
13. energy_above_hull；
14. density。

固定顺序便于可复现过滤轨迹。

## 9.3 统一原因码

```text
deprecated_material
duplicate_material_id
contains_excluded_element
missing_required_element
chemsys_mismatch
formula_mismatch
metallicity_missing
metallicity_mismatch
stability_missing
stability_mismatch
theoretical_flag_missing
theoretical_flag_mismatch
crystal_system_missing
crystal_system_mismatch
spacegroup_missing
spacegroup_mismatch
band_gap_missing
band_gap_below_min
band_gap_above_max
energy_above_hull_missing
energy_above_hull_below_min
energy_above_hull_above_max
density_missing
density_below_min
density_above_max
```

不要把原因写成任意自然语言字符串。报告层再将原因码映射为中文。

## 9.4 去重

第一阶段只按：

```text
(source, material_id)
```

去重。

若重复：

- 保留第一次出现；
- 后续记录写入 rejection；
- 不按化学式去重；
- 不尝试结构等价判断。

---

# 10. 排名算法

## 10.1 权重

第一阶段固定：

```python
STABILITY_WEIGHT = 0.45
BAND_GAP_WEIGHT = 0.40
COMPLETENESS_WEIGHT = 0.10
DIRECT_GAP_WEIGHT = 0.05
```

总和必须为 1。

## 10.2 稳定性得分

若请求提供 hull 上限 `H`：

```python
score = 1 - clamp(e_hull / H, 0, 1)
```

特殊情况：

- `H == 0` 时，`e_hull <= 1e-8` 得 1，否则得 0；
- 缺失值不应进入排名，因为已经在硬过滤剔除；
- 浮点最终 clamp 到 `[0, 1]`。

若请求没有 hull 条件：

```python
score = exp(-e_hull / 0.05)
```

若 hull 缺失：

```text
score = 0
```

## 10.3 带隙得分

目标值优先级：

1. 使用 `target_band_gap_ev`；
2. 否则若带隙上下限都存在，使用中点；
3. 否则若仅有下限，目标设为下限；
4. 否则若仅有上限，目标设为上限；
5. 否则 `band_gap_match=0.5`，表示没有用户目标。

容忍尺度：

```text
tolerance = max(区间宽度 / 2, 0.25 eV)
```

得分：

```python
score = max(0, 1 - abs(band_gap - target) / tolerance)
```

## 10.4 完整度

检查：

```text
band_gap_ev
energy_above_hull_ev_atom
formation_energy_ev_atom
density_g_cm3
symmetry
structure_dict
```

```python
score = present_count / 6
```

## 10.5 直接带隙

```text
True  -> 1.0
False -> 0.0
None  -> 0.0
```

第一阶段仅作为小权重偏好。

## 10.6 总分

```python
total_score = round(
    0.45 * stability + 0.40 * band_gap_match + 0.10 * completeness + 0.05 * direct_gap,
    8,
)
```

## 10.7 稳定排序

排序 key：

```python
(
    -total_score,
    energy_above_hull_or_inf,
    abs(band_gap - target_or_band_gap),
    material_id,
)
```

相同输入和相同数据库返回集合必须产生相同顺序。

---

# 11. 验证器

验证器在导出前运行。

## 11.1 必须验证

对每个候选重新验证：

- ID 唯一；
- 不 deprecated；
- 不含排除元素；
- 包含必需元素；
- 数值范围满足；
- bool 条件满足；
- rank 从 1 连续递增；
- 总分递减；
- 总分处于 0–1；
- score breakdown 重新计算一致；
- 关键属性 provenance 存在；
- 最终数量不超过 limit。

## 11.2 错误与警告

错误导致整个任务失败：

```text
候选违反硬约束
重复 ID
排名错误
总分计算不一致
属性没有来源
输出数量超过 limit
```

警告不导致失败：

```text
无结构，无法导出 CIF
形成能缺失
密度缺失
非直接带隙
理论材料
数据库版本不可获取
```

## 11.3 来源验证

若 `band_gap_ev` 不为空，则必须存在：

```python
provenance.property_name == "band_gap_ev"
```

其他属性同理。

Reporter 不得添加数据库未提供的数值。

---

# 12. ExportService

## 12.1 输出目录

```text
data/exports/<run_id>/
```

`run_id` 格式：

```text
run_YYYYMMDD_HHMMSS_<8位随机hex>
```

## 12.2 原子写入

所有文本输出：

1. 先写临时文件；
2. `flush`；
3. 使用 `Path.replace()` 原子替换；
4. 失败时清理临时文件。

## 12.3 JSON

`result.json` 保存完整 `ScreeningResult`。

使用：

```python
model_dump(mode="json")
```

确保 datetime 可序列化。

## 12.4 CSV

列：

```text
rank
material_id
formula_pretty
elements
chemsys
band_gap_ev
energy_above_hull_ev_atom
formation_energy_ev_atom
density_g_cm3
crystal_system
spacegroup_symbol
spacegroup_number
is_metal
is_gap_direct
is_stable
theoretical
total_score
stability_score
band_gap_match_score
completeness_score
direct_gap_score
source
database_version
```

使用 UTF-8 with BOM：

```python
encoding = "utf-8-sig"
```

便于 Windows Excel 打开中文表头或内容。

## 12.5 CIF

```python
from pymatgen.core import Structure
```

规则：

- 只导出存在 structure 的候选；
- 文件名只使用 `material_id` 经白名单清洗；
- 允许字符 `[A-Za-z0-9._-]`；
- 目录固定为 run 目录下 `cif/`；
- 路径 resolve 后必须仍位于 export root；
- 无结构产生 warning，不导致整个任务失败。

## 12.6 Markdown 报告

报告只使用模板，不用 LLM。

固定章节：

```text
# 材料筛选结果
## 筛选条件
## 数据来源
## 筛选统计
## 候选材料
## 排名方法
## 验证结果
## 警告
## 科学说明
```

固定科学说明：

```text
本结果主要基于 Materials Project 中的计算数据。数据库中的带隙、
能量高于凸包、形成能等属性不能直接等同于实验测量值。较低的
能量高于凸包和较高的筛选排名不表示材料必然可合成、无毒、
长期稳定或适用于实际器件，仍需结合更高精度计算、文献和实验验证。
```

---

# 13. ScreeningService

```python
class ScreeningService:
    def __init__(
        self,
        repository: MaterialsRepository,
        filter_service: FilterService,
        ranking_service: RankingService,
        validation_service: ValidationService,
        export_service: ExportService,
    ) -> None: ...

    def run(
        self,
        request: ScreeningRequest,
        output_root: Path,
    ) -> ScreeningRunOutput:
        started_at = utc_now()
        retrieval = self._repository.search(request)

        filtered, filter_trace = self._filter_service.apply(
            retrieval.records,
            request,
        )

        ranked_all = self._ranking_service.rank(filtered, request)
        ranked_limited = ranked_all[: request.limit]

        result = build_result(...)
        validation = self._validation_service.validate(result)

        if not validation.passed:
            raise ValidationFailedError(...)

        result = result.model_copy(update={"validation": validation})

        exports = self._export_service.export(result, output_root)
        return ScreeningRunOutput(result=result, exports=exports)
```

注意：

- 先对全部通过候选排序，再截断；
- 截断后验证；
- 导出失败不应留下伪装完整的结果目录；
- service 不捕获所有异常，CLI 负责转成用户信息。

---

# 14. CLI

使用 Typer。

## 14.1 命令

```bash
materials-screen validate-request --request path.json
materials-screen screen --request path.json --output data/exports
materials-screen inspect-mp
materials-screen version
```

## 14.2 `validate-request`

仅解析和验证请求，不访问网络。

成功：

```text
Request is valid
Fingerprint: ...
```

失败：

```text
Request is invalid
- band_gap_ev.min: must be greater than or equal to 0
```

## 14.3 `screen`

参数：

```text
--request / -r   必填 JSON
--output / -o    默认 data/exports
--repository     materials-project 或 mock，默认 materials-project
--fixture        使用 mock 时的 fixture
--no-cif         可选
```

## 14.4 终端输出

使用 Rich 表格，但不要把终端表格作为唯一输出。

最多显示前 20 项。

---

# 15. 查询指纹

```python
def request_fingerprint(request: ScreeningRequest) -> str:
    canonical = orjson.dumps(
        request.model_dump(mode="json", exclude_none=True),
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(canonical).hexdigest()
```

测试：

- 字段顺序不影响；
- 相同元素不同输入顺序在标准化后不影响；
- 不同阈值产生不同指纹。

---

# 16. 测试设计

## 16.1 测试禁令

普通测试不得：

- 访问互联网；
- 使用真实 API Key；
- 依赖当前数据库内容；
- 依赖本地时区；
- 依赖随机排序；
- 写入仓库外任意路径。

## 16.2 时间

通过注入 clock 或固定测试时间，使 metadata 可断言。

## 16.3 单元测试清单

### models

- 合法范围；
- min > max；
- 两个 bound 都空；
- 负带隙；
- 负 hull；
- 非法元素；
- required/excluded 冲突；
- 非法空间群；
- target 不在范围；
- extra 字段拒绝。

### chemistry

- `pb` -> `Pb`；
- `PB` -> `Pb`；
- `Li-Fe-O` 标准化；
- 重复元素去重；
- 非法 `Xx` 报错。

### MP mapping

- SummaryDoc-like object；
- dict；
- structure dict；
- structure object；
- symmetry dict；
- symmetry object；
- 缺失字段；
- 非法结构字段；
- provenance 数量和内容。

### filters

每一个 rejection code 至少一个测试。

### ranking

- 分数边界；
- hull=0；
- target 选择规则；
- 完整度；
- 直接带隙；
- tie-break；
- 输入顺序变化不影响最终排序。

### validator

- 正常通过；
- 违反元素条件；
- 数值越界；
- 重复 ID；
- rank 不连续；
- 分数错误；
- provenance 缺失；
- limit 超限。

### export

- JSON round trip；
- CSV 列；
- UTF-8 BOM；
- CIF 正常；
- 无 structure；
- 非法 material_id 清洗；
- 路径穿越阻止；
- 临时目录清理。

## 16.4 Mock 集成测试

完整调用：

```text
JSON request
  → ScreeningRequest
  → MockRepository
  → filter
  → rank
  → validate
  → export
```

断言：

- 通过数正确；
- 排名正确；
- 导出文件正确；
- report 有免责声明；
- 没有网络调用。

## 16.5 CLI 测试

使用 Typer `CliRunner`：

- 正常 mock 命令；
- 请求文件不存在；
- 请求非法；
- 未配置 API Key；
- 导出目录不可写；
- 退出码正确。

## 16.6 手动真实 API 冒烟测试

文件：

```text
tests/manual/test_real_mp_api.py
```

标记：

```python
@pytest.mark.real_api
```

只有：

```bash
RUN_REAL_MP_TESTS=1
MP_API_KEY=...
uv run pytest -m real_api -q
```

才执行。

冒烟查询必须小，例如固定材料 ID 或窄化学体系，避免大规模下载。

## 16.7 质量门

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not real_api" --cov=materials_screening --cov-report=term-missing
```

目标：

```text
总覆盖率 >= 90%
领域层和服务层 >= 95%
```

---

# 17. README 必须写明

1. 项目目的；
2. 当前仅为第一阶段纯规则版本；
3. 安装步骤；
4. 获取 Materials Project API Key 的说明；
5. `.env` 配置；
6. Mock 模式运行；
7. 真实模式运行；
8. 输入 JSON Schema；
9. 输出文件；
10. 测试命令；
11. 依赖版本查看；
12. 科学免责声明；
13. 常见错误；
14. 不支持自然语言输入。

---

# 18. `scripts/inspect_mp_fields.py`

要求代码智能体实现：

```python
import inspect
from importlib.metadata import version

from mp_api.client import MPRester


def main() -> None:
    print("mp-api:", version("mp-api"))
    print("pymatgen:", version("pymatgen"))

    signature = inspect.signature(type(MPRester().materials.summary).search)
    print("Summary search signature:")
    print(signature)

    # 需要 Key 的字段或 DB version 检查应明确处理
```

更稳妥的实现不要在没有 Key 时创建真实客户端。可以：

- 从 `SummaryRester.search` 类方法检查签名；
- 若有 Key，再创建 `MPRester` 输出 available_fields 和 DB version。

---

# 19. AGENTS/CLAUDE 工作方式

仓库根目录放：

```text
AGENTS.md
CLAUDE.md
```

Codex 使用 `AGENTS.md`。

Claude Code 的 `CLAUDE.md` 内容：

```markdown
@AGENTS.md

## Claude Code 补充规则

- 每次只完成用户当前指定的里程碑。
- 修改前先读取相关测试和接口。
- 不要自动运行真实 Materials Project API 测试。
```

持久指令必须短、明确、可验证；详细实现内容保存在本技术文档中，由实时提示词要求工具按需阅读。

---

# 20. 里程碑与验收

## M1 工程骨架

完成：

- pyproject；
- src layout；
- CLI version；
- lint/test 配置；
- AGENTS/CLAUDE；
- README 骨架。

验收：

```bash
uv sync
uv run materials-screen version
uv run ruff check .
uv run mypy src
uv run pytest
```

## M2 模型和化学标准化

完成全部 Pydantic 模型和校验。

验收：

```bash
uv run pytest tests/unit/test_models.py tests/unit/test_chemistry.py
```

## M3 Repository

完成 Protocol、Mock、MP 查询参数和映射。

验收：

```bash
uv run pytest tests/unit/test_mp_mapping.py
```

普通测试不联网。

## M4 过滤和排名

完成硬过滤、轨迹、可解释排名。

验收：

```bash
uv run pytest tests/unit/test_filter_service.py tests/unit/test_ranking_service.py
```

## M5 验证和导出

完成验证器、JSON/CSV/CIF/Markdown。

验收：

```bash
uv run pytest tests/unit/test_validation_service.py tests/unit/test_export_service.py
```

## M6 CLI 和集成

完成完整 mock 流程。

验收：

```bash
uv run pytest tests/integration
uv run materials-screen screen \
  --repository mock \
  --fixture tests/fixtures/mp_documents.json \
  --request examples/semiconductor_request.json
```

## M7 真实 API 冒烟和收尾

手动执行：

```bash
RUN_REAL_MP_TESTS=1 uv run pytest -m real_api -q
```

最后质量门全部通过。

---

# 21. 第一阶段“完成”的严格定义

只有全部满足才算完成：

- [ ] 不使用 LLM；
- [ ] 不使用 LangGraph；
- [ ] 结构化请求可验证；
- [ ] 元素符号和范围规则正确；
- [ ] MP Repository 可真实查询；
- [ ] Mock Repository 可离线测试；
- [ ] 所有硬约束本地再次执行；
- [ ] 缺失硬约束属性时拒绝；
- [ ] 排名确定且有分解；
- [ ] 所有候选均经过 Validator；
- [ ] 数据属性具有 provenance；
- [ ] JSON、CSV、CIF、Markdown 可导出；
- [ ] 无结构不会导致整个任务失败；
- [ ] API Key 不进入仓库和日志；
- [ ] 普通测试不联网；
- [ ] 真实 API 测试默认跳过；
- [ ] 总覆盖率达到目标；
- [ ] Ruff、mypy、pytest 全部通过；
- [ ] Windows 和 Linux 路径使用 `pathlib`；
- [ ] README 可让新用户独立运行。

---

# 22. 官方资料核对清单

实施时优先查阅：

- Materials Project API 使用说明；
- `mp-api` SummaryRester 当前签名；
- Materials Project database version；
- pymatgen 当前文档；
- Pydantic v2；
- uv；
- Typer；
- Codex `AGENTS.md` 指令机制；
- Claude Code `CLAUDE.md` 指令机制。

本阶段不要根据博客或旧代码猜测 Materials Project 字段。
