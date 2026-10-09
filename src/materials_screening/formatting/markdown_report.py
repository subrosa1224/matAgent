"""Fixed-template Markdown report for screening results (M5)."""

import json

from materials_screening.models import ScreeningResult
from materials_screening.services.ranking_service import (
    BAND_GAP_WEIGHT,
    COMPLETENESS_WEIGHT,
    DIRECT_GAP_WEIGHT,
    STABILITY_WEIGHT,
)

DISCLAIMER = (
    "本结果主要基于 Materials Project 中的计算数据。数据库中的带隙、"
    "能量高于凸包、形成能等属性不能直接等同于实验测量值。较低的"
    "能量高于凸包和较高的筛选排名不表示材料必然可合成、无毒、"
    "长期稳定或适用于实际器件，仍需结合更高精度计算、文献和实验验证。"
)


def _fmt(value: object) -> str:
    return "" if value is None else str(value)


def build_markdown_report(result: ScreeningResult) -> str:
    """Build the fixed-template report from a ScreeningResult only."""
    lines: list[str] = ["# 材料筛选结果", ""]

    lines.append("## 筛选条件")
    lines.append("")
    request_json = json.dumps(
        result.request.model_dump(mode="json", exclude_none=True),
        indent=2,
        ensure_ascii=False,
    )
    lines.append("```json")
    lines.append(request_json)
    lines.append("```")
    lines.append("")

    metadata = result.metadata
    lines.append("## 数据来源")
    lines.append("")
    lines.append(f"- source: {metadata.source}")
    lines.append(f"- database_version: {metadata.database_version or '不可用'}")
    lines.append(f"- mp_api_version: {metadata.mp_api_version or '不可用'}")
    lines.append(f"- pymatgen_version: {metadata.pymatgen_version or '不可用'}")
    lines.append(f"- application_version: {metadata.application_version}")
    lines.append(f"- query_fingerprint: {metadata.query_fingerprint}")
    lines.append("")

    lines.append("## 筛选统计")
    lines.append("")
    lines.append(f"- retrieved_count: {result.retrieved_count}")
    lines.append(f"- passed_filter_count: {result.passed_filter_count}")
    lines.append(f"- returned: {len(result.ranked_materials)}")
    lines.append("")

    lines.append("## 候选材料")
    lines.append("")
    lines.append(
        "| 排名 | material_id | formula_pretty | band_gap_ev | "
        "energy_above_hull_ev_atom | total_score |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in result.ranked_materials:
        record = item.record
        lines.append(
            f"| {item.rank} | {record.material_id} | "
            f"{record.formula_pretty} | {_fmt(record.band_gap_ev)} | "
            f"{_fmt(record.energy_above_hull_ev_atom)} | "
            f"{item.total_score} |"
        )
    lines.append("")

    lines.append("## 排名方法")
    lines.append("")
    lines.append(
        "第一阶段使用固定权重："
        f"稳定性 {STABILITY_WEIGHT}、带隙匹配 {BAND_GAP_WEIGHT}、"
        f"完整度 {COMPLETENESS_WEIGHT}、直接带隙 {DIRECT_GAP_WEIGHT}；"
        "总分为加权和并保留 8 位精度；同分时按能量高于凸包、带隙距离"
        "目标值和 material_id 稳定排序。"
    )
    lines.append("")

    validation = result.validation
    lines.append("## 验证结果")
    lines.append("")
    lines.append(f"- passed: {'是' if validation.passed else '否'}")
    lines.append(f"- errors: {len(validation.errors)}")
    lines.append(f"- warnings: {len(validation.warnings)}")
    lines.append("")

    lines.append("## 警告")
    lines.append("")
    if validation.warnings:
        for warning in validation.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("无")
    lines.append("")

    lines.append("## 科学说明")
    lines.append("")
    lines.append(DISCLAIMER)
    lines.append("")
    return "\n".join(lines)
