"""Property definitions that bound scientifically supported screening behavior."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.research.errors import ResearchProjectError
from materials_screening.research.models import (
    CriterionOperator,
    CriterionRole,
    MissingValuePolicy,
    SourceRole,
)


class PropertyValueType(StrEnum):
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    TEXT = "text"
    TEXT_SET = "text_set"


class PropertyDefinition(BaseModel):
    """A canonical property and the exact ways it may be used."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    property_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    label_zh: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    canonical_unit: str | None = None
    value_type: PropertyValueType
    supported_roles: frozenset[CriterionRole]
    supported_operators: frozenset[CriterionOperator]
    source_roles: frozenset[SourceRole]
    allowed_missing_policies: frozenset[MissingValuePolicy]
    query_supported: bool
    comparability_notes: str = Field(min_length=1)


def _key(value: str) -> str:
    return "_".join(value.strip().casefold().replace("-", " ").split())


class PropertyRegistry:
    """Immutable lookup table with explicit, collision-free aliases."""

    def __init__(self, definitions: tuple[PropertyDefinition, ...]) -> None:
        if not definitions:
            raise ResearchProjectError("EMPTY_PROPERTY_REGISTRY", "registry is empty")
        by_id: dict[str, PropertyDefinition] = {}
        aliases: dict[str, str] = {}
        for definition in definitions:
            if definition.property_id in by_id:
                raise ResearchProjectError(
                    "DUPLICATE_PROPERTY",
                    f"duplicate property: {definition.property_id}",
                )
            by_id[definition.property_id] = definition
            names = (definition.property_id, definition.label_zh, *definition.aliases)
            for name in names:
                key = _key(name)
                owner = aliases.get(key)
                if owner is not None and owner != definition.property_id:
                    raise ResearchProjectError(
                        "ALIAS_COLLISION",
                        f"property alias {name!r} belongs to both {owner!r} and "
                        f"{definition.property_id!r}",
                    )
                aliases[key] = definition.property_id
        self._by_id = by_id
        self._aliases = aliases

    @property
    def definitions(self) -> tuple[PropertyDefinition, ...]:
        return tuple(self._by_id.values())

    def resolve(self, name: str) -> PropertyDefinition | None:
        property_id = self._aliases.get(_key(name))
        return None if property_id is None else self._by_id[property_id]

    def require(self, name: str) -> PropertyDefinition:
        definition = self.resolve(name)
        if definition is None:
            raise ResearchProjectError(
                "UNREGISTERED_PROPERTY", f"property is not registered: {name!r}"
            )
        return definition


_FILTER_RANK_REPORT = frozenset(CriterionRole)
_FILTER_REPORT = frozenset({CriterionRole.HARD_FILTER, CriterionRole.REPORT_ONLY})
_REPORT_ONLY = frozenset({CriterionRole.REPORT_ONLY})
_NUMERIC_OPERATORS = frozenset(
    {
        CriterionOperator.EQ,
        CriterionOperator.GTE,
        CriterionOperator.LTE,
        CriterionOperator.BETWEEN,
        CriterionOperator.TARGET,
        CriterionOperator.PREFER_MIN,
        CriterionOperator.PREFER_MAX,
    }
)
_EQUALITY = frozenset({CriterionOperator.EQ})
_CALCULATED = frozenset({SourceRole.DATABASE_CALCULATED})
_DATABASE_METADATA = frozenset({SourceRole.DATABASE_METADATA})
_STANDARD_MISSING = frozenset(
    {
        MissingValuePolicy.EXCLUDE,
        MissingValuePolicy.KEEP_WITH_WARNING,
        MissingValuePolicy.FAIL_PROJECT,
    }
)


def _numeric(
    property_id: str,
    label_zh: str,
    unit: str,
    aliases: tuple[str, ...],
    notes: str,
) -> PropertyDefinition:
    return PropertyDefinition(
        property_id=property_id,
        label_zh=label_zh,
        aliases=aliases,
        canonical_unit=unit,
        value_type=PropertyValueType.NUMBER,
        supported_roles=_FILTER_RANK_REPORT,
        supported_operators=_NUMERIC_OPERATORS,
        source_roles=_CALCULATED,
        allowed_missing_policies=_STANDARD_MISSING,
        query_supported=True,
        comparability_notes=notes,
    )


def default_property_registry() -> PropertyRegistry:
    """Return the conservative RA-1 registry backed by current MP records."""
    definitions = (
        _numeric(
            "band_gap_ev",
            "带隙",
            "eV",
            ("band gap", "band_gap", "禁带宽度"),
            "数据库计算带隙；不能直接替代实验光学带隙或击穿强度。",
        ),
        _numeric(
            "energy_above_hull_ev_atom",
            "凸包上方能量",
            "eV/atom",
            ("energy above hull", "e_above_hull", "热力学亚稳程度"),
            "计算热力学指标；不等同于不可合成，也不证明动力学稳定性。",
        ),
        _numeric(
            "formation_energy_ev_atom",
            "形成能",
            "eV/atom",
            ("formation energy", "形成能每原子"),
            "仅在计算口径和参考态一致时进行精确数值比较。",
        ),
        _numeric(
            "density_g_cm3",
            "密度",
            "g/cm^3",
            ("density", "density_g_cm3"),
            "数据库晶体密度；与多孔、薄膜或非化学计量样品不可直接等同。",
        ),
        PropertyDefinition(
            property_id="is_stable",
            label_zh="数据库稳定标记",
            aliases=("stable", "是否稳定"),
            value_type=PropertyValueType.BOOLEAN,
            supported_roles=_FILTER_REPORT,
            supported_operators=_EQUALITY,
            source_roles=_CALCULATED,
            allowed_missing_policies=_STANDARD_MISSING,
            query_supported=True,
            comparability_notes="数据库计算标签；False 不表示实验上不能合成。",
        ),
        PropertyDefinition(
            property_id="is_metal",
            label_zh="金属性标记",
            aliases=("metallic", "是否金属"),
            value_type=PropertyValueType.BOOLEAN,
            supported_roles=_FILTER_REPORT,
            supported_operators=_EQUALITY,
            source_roles=_CALCULATED,
            allowed_missing_policies=_STANDARD_MISSING,
            query_supported=True,
            comparability_notes="数据库电子结构分类，依赖所用计算方法。",
        ),
        PropertyDefinition(
            property_id="is_gap_direct",
            label_zh="直接带隙标记",
            aliases=("direct gap", "是否直接带隙"),
            value_type=PropertyValueType.BOOLEAN,
            supported_roles=_FILTER_REPORT,
            supported_operators=_EQUALITY,
            source_roles=_CALCULATED,
            allowed_missing_policies=_STANDARD_MISSING,
            query_supported=False,
            comparability_notes="快照中有该字段，但当前生产查询合同尚不能执行硬过滤。",
        ),
        PropertyDefinition(
            property_id="theoretical",
            label_zh="理论结构标记",
            aliases=("is theoretical", "是否理论结构"),
            value_type=PropertyValueType.BOOLEAN,
            supported_roles=_FILTER_REPORT,
            supported_operators=_EQUALITY,
            source_roles=_DATABASE_METADATA,
            allowed_missing_policies=_STANDARD_MISSING,
            query_supported=True,
            comparability_notes="来源状态字段，不构成已合成或未合成的独立证明。",
        ),
        PropertyDefinition(
            property_id="spacegroup_number",
            label_zh="空间群编号",
            aliases=("space group", "spacegroup", "空间群"),
            value_type=PropertyValueType.INTEGER,
            supported_roles=_FILTER_REPORT,
            supported_operators=_EQUALITY,
            source_roles=_DATABASE_METADATA,
            allowed_missing_policies=_STANDARD_MISSING,
            query_supported=True,
            comparability_notes="空间群相同不足以证明结构身份相同，仍需组成和结构核验。",
        ),
        PropertyDefinition(
            property_id="crystal_system",
            label_zh="晶系",
            aliases=("crystal system",),
            value_type=PropertyValueType.TEXT,
            supported_roles=_FILTER_REPORT,
            supported_operators=_EQUALITY,
            source_roles=_DATABASE_METADATA,
            allowed_missing_policies=_STANDARD_MISSING,
            query_supported=True,
            comparability_notes="晶系只用于粗粒度结构筛选，不能替代物相身份判断。",
        ),
        PropertyDefinition(
            property_id="formula_pretty",
            label_zh="化学式",
            aliases=("formula", "composition", "组成"),
            value_type=PropertyValueType.TEXT,
            supported_roles=_REPORT_ONLY,
            supported_operators=_EQUALITY,
            source_roles=_DATABASE_METADATA,
            allowed_missing_policies=frozenset({MissingValuePolicy.NOT_APPLICABLE}),
            query_supported=True,
            comparability_notes="化学式相同不代表晶体物相或结构记录相同。",
        ),
        PropertyDefinition(
            property_id="elements",
            label_zh="元素集合",
            aliases=("element set", "元素"),
            value_type=PropertyValueType.TEXT_SET,
            supported_roles=_REPORT_ONLY,
            supported_operators=_EQUALITY,
            source_roles=_DATABASE_METADATA,
            allowed_missing_policies=frozenset({MissingValuePolicy.NOT_APPLICABLE}),
            query_supported=True,
            comparability_notes="元素集合用于描述组成范围，不代表化学计量或物相身份。",
        ),
    )
    return PropertyRegistry(definitions)
