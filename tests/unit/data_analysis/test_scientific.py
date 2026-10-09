from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.scientific import (
    ScientificAnalysisBrief,
    ScientificBriefService,
)


def test_materials_role_proposal_is_useful_but_never_confirmed(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "store")
    dataset = store.register_file(
        Path("examples/data_analysis/grouped_experiment_demo.csv"),
        source_artifact_id="artifact-scientific-proposal",
    )

    proposal = ScientificBriefService(store).propose_roles(dataset.dataset_id)

    assert proposal.requires_user_confirmation is True
    assert proposal.group_candidates[0] == "method"
    assert "hardness_hv" in proposal.response_candidates[:4]
    assert "sample_id" in proposal.identifier_candidates
    assert any("用户确认" in item for item in proposal.rationale)


def test_unconfirmed_brief_cannot_authorize_inferential_analysis() -> None:
    brief = ScientificAnalysisBrief(
        dataset_id="dataset-scientific",
        domain="materials_science",
        research_question="不同烧结方法是否影响材料硬度？",
        observation_unit="独立材料试样",
        design="independent_groups",
        response_variables=("hardness_hv",),
        group_variable="method",
        roles_confirmed=False,
    )

    with pytest.raises(ValueError, match="must be confirmed"):
        ScientificBriefService.require_confirmed(brief)


def test_confirmed_brief_enforces_design_and_variable_role_boundaries() -> None:
    brief = ScientificAnalysisBrief(
        dataset_id="dataset-scientific",
        domain="materials_science",
        research_question="不同烧结方法是否产生具有实际意义的硬度差异？",
        observation_unit="独立材料试样",
        design="independent_groups",
        response_variables=("hardness_hv",),
        group_variable="method",
        hypothesis="至少一种方法的平均硬度不同",
        practical_thresholds={"hardness_hv": 20.0},
        units={"hardness_hv": "HV"},
        roles_confirmed=True,
    )
    ScientificBriefService.require_confirmed(brief)

    with pytest.raises(ValidationError, match="requires a subject ID"):
        ScientificAnalysisBrief(
            dataset_id="dataset-scientific",
            research_question="同一样品处理前后硬度是否变化？",
            observation_unit="材料试样",
            design="paired",
            response_variables=("hardness_after",),
            roles_confirmed=True,
        )

    with pytest.raises(ValidationError, match="requires a condition variable"):
        ScientificAnalysisBrief(
            dataset_id="dataset-scientific",
            research_question="同一样品处理前后硬度是否变化？",
            observation_unit="材料试样",
            design="paired",
            response_variables=("hardness_hv",),
            subject_id_variable="sample_id",
            roles_confirmed=True,
        )

    with pytest.raises(ValidationError, match="requires a predictor"):
        ScientificAnalysisBrief(
            dataset_id="dataset-scientific",
            research_question="温度与材料硬度是否存在线性关系？",
            observation_unit="独立材料试样",
            design="continuous_relationship",
            response_variables=("hardness_hv",),
            roles_confirmed=True,
        )
