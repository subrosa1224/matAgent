from __future__ import annotations

from pathlib import Path

import pytest

from materials_screening.research import (
    CandidateDecision,
    CandidateEvidenceLedger,
    CandidateSourceKind,
    CandidateTier,
    ClaimEvidenceLink,
    ClaimKind,
    EvidenceLocator,
    EvidenceReviewStatus,
    EvidenceSourceKind,
    EvidenceStance,
    EvidenceStatus,
    HardConstraintStatus,
    IdentityRelation,
    IdentityUse,
    MaterialCandidate,
    MaterialIdentityAssessmentRecord,
    MaterialIdentityRecord,
    ProjectConflictError,
    PropertyValue,
    ResearchProjectError,
    ScreeningClaim,
    ScreeningEvidence,
    SourceRole,
)


def _database_evidence(
    *,
    evidence_id: str = "evidence-band-gap",
    candidate_id: str = "candidate-1",
    value: float = 3.2,
    status: EvidenceReviewStatus = EvidenceReviewStatus.VALIDATED,
) -> ScreeningEvidence:
    return ScreeningEvidence(
        evidence_id=evidence_id,
        project_id="project-1",
        candidate_id=candidate_id,
        source_kind=EvidenceSourceKind.MATERIALS_DATABASE,
        source_id="query-1",
        locator=EvidenceLocator(
            query_id="query-1",
            source_material_id="mp-1",
            database_version="2026.04.13",
        ),
        property_id="band_gap_ev",
        value=value,
        unit="eV",
        method="database summary",
        review_status=status,
    )


def _candidate(
    *,
    candidate_id: str = "candidate-1",
    evidence_id: str = "evidence-band-gap",
    value: float = 3.2,
) -> MaterialCandidate:
    return MaterialCandidate(
        candidate_id=candidate_id,
        project_id="project-1",
        source_kind=CandidateSourceKind.MATERIALS_DATABASE,
        source_material_id="mp-1",
        formula="TiO2",
        chemsys="O-Ti",
        space_group_number=136,
        property_values=(
            PropertyValue(
                property_id="band_gap_ev",
                value=value,
                unit="eV",
                source_role=SourceRole.DATABASE_CALCULATED,
                evidence_id=evidence_id,
            ),
        ),
        provenance={"query_id": "query-1"},
    )


def _literature_evidence() -> ScreeningEvidence:
    return ScreeningEvidence(
        evidence_id="evidence-literature",
        project_id="project-1",
        candidate_id="candidate-1",
        source_kind=EvidenceSourceKind.LITERATURE_FULL_TEXT,
        source_id="doc-1",
        locator=EvidenceLocator(
            document_id="doc-1",
            chunk_id="chunk-1",
            page_from=2,
            page_to=2,
        ),
        property_id="band_gap_ev",
        value=3.3,
        unit="eV",
        source_excerpt="The measured band gap was 3.3 eV.",
        source_text_sha256="a" * 64,
        review_status=EvidenceReviewStatus.HUMAN_REVIEWED,
    )


def test_candidate_bundle_and_records_are_immutable(tmp_path: Path) -> None:
    ledger = CandidateEvidenceLedger(tmp_path / "ledger")
    evidence = _database_evidence()
    candidate = _candidate()

    ledger.add_candidate_bundle(candidate, (evidence,))
    ledger.add_candidate_bundle(candidate, (evidence,))

    assert ledger.get_candidate("project-1", "candidate-1") == candidate
    assert ledger.get_evidence("project-1", "evidence-band-gap") == evidence
    decision = CandidateDecision(
        decision_id="decision-1",
        project_id="project-1",
        candidate_id="candidate-1",
        hard_constraint_status=HardConstraintStatus.PASSED,
        evidence_status=EvidenceStatus.INSUFFICIENT,
        evidence_ids=("evidence-band-gap",),
    )
    ledger.add_decision(decision)
    assert ledger.get_decision("project-1", "decision-1") == decision
    with pytest.raises(ProjectConflictError) as exc_info:
        ledger.add_candidate_bundle(
            candidate.model_copy(update={"formula": "SnO2"}),
            (evidence,),
        )
    assert exc_info.value.code == "IMMUTABLE_LEDGER_RECORD"


def test_candidate_property_must_match_structured_evidence(tmp_path: Path) -> None:
    ledger = CandidateEvidenceLedger(tmp_path / "ledger")

    with pytest.raises(ResearchProjectError) as exc_info:
        ledger.add_candidate_bundle(
            _candidate(value=3.1),
            (_database_evidence(value=3.2),),
        )

    assert exc_info.value.code == "PROPERTY_EVIDENCE_MISMATCH"


def test_pending_evidence_cannot_support_claim(tmp_path: Path) -> None:
    ledger = CandidateEvidenceLedger(tmp_path / "ledger")
    evidence = _database_evidence(status=EvidenceReviewStatus.PENDING)
    candidate = _candidate()
    ledger.add_candidate_bundle(candidate, (evidence,))
    claim = ScreeningClaim(
        claim_id="claim-1",
        project_id="project-1",
        candidate_id="candidate-1",
        kind=ClaimKind.PROPERTY_VALUE,
        property_id="band_gap_ev",
        statement="该候选的计算带隙为 3.2 eV。",
    )
    ledger.add_claim(claim)

    with pytest.raises(ResearchProjectError) as exc_info:
        ledger.add_link(
            ClaimEvidenceLink(
                link_id="link-1",
                project_id="project-1",
                claim_id="claim-1",
                evidence_id="evidence-band-gap",
                stance=EvidenceStance.SUPPORTS,
                explanation="direct property evidence",
            )
        )

    assert exc_info.value.code == "UNVALIDATED_EVIDENCE_LINK"


def test_formula_only_background_cannot_support_candidate_claim(tmp_path: Path) -> None:
    ledger = CandidateEvidenceLedger(tmp_path / "ledger")
    evidence = _database_evidence()
    ledger.add_candidate_bundle(_candidate(), (evidence,))
    literature = _literature_evidence()
    ledger.add_evidence(literature)
    ledger.add_identity(
        MaterialIdentityAssessmentRecord(
            identity_id="identity-1",
            project_id="project-1",
            candidate_id="candidate-1",
            evidence_id="evidence-literature",
            left=MaterialIdentityRecord(
                source_kind="materials_project",
                source_material_id="mp-1",
                formula="TiO2",
            ),
            right=MaterialIdentityRecord(
                source_kind="literature",
                source_material_id="sample-1",
                formula="TiO2",
            ),
            relation=IdentityRelation.SAME_FORMULA_INCOMPLETE_STRUCTURE,
            identity_use=IdentityUse.BACKGROUND_ONLY,
            same_structure_candidate=False,
            requires_human_review=True,
            reasons=("formula only",),
            review_status="pending_domain_review",
        )
    )
    ledger.add_claim(
        ScreeningClaim(
            claim_id="claim-1",
            project_id="project-1",
            candidate_id="candidate-1",
            kind=ClaimKind.PROPERTY_VALUE,
            property_id="band_gap_ev",
            statement="candidate property claim",
        )
    )

    with pytest.raises(ResearchProjectError) as exc_info:
        ledger.add_link(
            ClaimEvidenceLink(
                link_id="link-1",
                project_id="project-1",
                claim_id="claim-1",
                evidence_id="evidence-literature",
                stance=EvidenceStance.SUPPORTS,
                identity_assessment_id="identity-1",
                explanation="formula-only mapping",
            )
        )

    assert exc_info.value.code == "IDENTITY_USE_BLOCKS_CLAIM"


def test_cross_candidate_evidence_cannot_enter_decision(tmp_path: Path) -> None:
    ledger = CandidateEvidenceLedger(tmp_path / "ledger")
    first_evidence = _database_evidence()
    ledger.add_candidate_bundle(_candidate(), (first_evidence,))
    second_evidence = _database_evidence(
        evidence_id="evidence-other",
        candidate_id="candidate-2",
    )
    ledger.add_candidate_bundle(
        _candidate(candidate_id="candidate-2", evidence_id="evidence-other"),
        (second_evidence,),
    )

    with pytest.raises(ResearchProjectError) as exc_info:
        ledger.add_decision(
            CandidateDecision(
                decision_id="decision-1",
                project_id="project-1",
                candidate_id="candidate-1",
                hard_constraint_status=HardConstraintStatus.PASSED,
                evidence_status=EvidenceStatus.SUPPORTED,
                final_tier=CandidateTier.B,
                evidence_ids=("evidence-other",),
            )
        )

    assert exc_info.value.code == "DECISION_EVIDENCE_CANDIDATE_MISMATCH"


def test_failed_candidate_requires_exclusion_reason_and_cannot_be_tier_a() -> None:
    with pytest.raises(ValueError, match="exclusion reason"):
        CandidateDecision(
            decision_id="decision-1",
            project_id="project-1",
            candidate_id="candidate-1",
            hard_constraint_status=HardConstraintStatus.FAILED,
        )

    with pytest.raises(ValueError, match="cannot receive"):
        CandidateDecision(
            decision_id="decision-1",
            project_id="project-1",
            candidate_id="candidate-1",
            hard_constraint_status=HardConstraintStatus.FAILED,
            exclusion_reasons=("band gap below threshold",),
            final_tier=CandidateTier.A,
        )


def test_full_text_evidence_requires_exact_locator_excerpt_and_hash() -> None:
    with pytest.raises(ValueError, match="document, chunk, and pages"):
        ScreeningEvidence(
            evidence_id="evidence-lit",
            project_id="project-1",
            source_kind=EvidenceSourceKind.LITERATURE_FULL_TEXT,
            source_id="doc-1",
            locator=EvidenceLocator(document_id="doc-1"),
            source_excerpt="reported text",
            source_text_sha256="a" * 64,
            review_status=EvidenceReviewStatus.PENDING,
        )


def test_unregistered_property_cannot_enter_ledger(tmp_path: Path) -> None:
    ledger = CandidateEvidenceLedger(tmp_path / "ledger")
    evidence = _database_evidence().model_copy(
        update={"property_id": "imaginary_application_score"}
    )

    with pytest.raises(ResearchProjectError) as exc_info:
        ledger.add_evidence(evidence)

    assert exc_info.value.code == "UNREGISTERED_PROPERTY_EVIDENCE"


def test_human_reviewed_structure_mapping_can_support_literature_claim(
    tmp_path: Path,
) -> None:
    ledger = CandidateEvidenceLedger(tmp_path / "ledger")
    database = _database_evidence()
    ledger.add_candidate_bundle(_candidate(), (database,))
    literature = _literature_evidence()
    ledger.add_evidence(literature)
    identity = MaterialIdentityAssessmentRecord(
        identity_id="identity-1",
        project_id="project-1",
        candidate_id="candidate-1",
        evidence_id="evidence-literature",
        left=MaterialIdentityRecord(
            source_kind="materials_project",
            source_material_id="mp-1",
            formula="TiO2",
            space_group_number=136,
            structure_fingerprint="same-structure",
        ),
        right=MaterialIdentityRecord(
            source_kind="literature",
            source_material_id="sample-1",
            formula="TiO2",
            space_group_number=136,
            structure_fingerprint="same-structure",
        ),
        relation=IdentityRelation.STRUCTURE_MATCH,
        identity_use=IdentityUse.CANDIDATE_PROPERTY,
        same_structure_candidate=True,
        requires_human_review=True,
        reasons=("structure mapping reviewed",),
        review_status="human_reviewed",
    )
    ledger.add_identity(identity)
    claim = ScreeningClaim(
        claim_id="claim-1",
        project_id="project-1",
        candidate_id="candidate-1",
        kind=ClaimKind.PROPERTY_VALUE,
        property_id="band_gap_ev",
        statement="该物相的实验带隙为 3.3 eV。",
    )
    ledger.add_claim(claim)
    link = ClaimEvidenceLink(
        link_id="link-1",
        project_id="project-1",
        claim_id="claim-1",
        evidence_id="evidence-literature",
        stance=EvidenceStance.SUPPORTS,
        identity_assessment_id="identity-1",
        explanation="reviewed phase mapping and full-text measurement",
    )

    assert ledger.add_link(link) == link
    assert ledger.get_identity("project-1", "identity-1") == identity
    assert ledger.get_link("project-1", "link-1") == link


@pytest.mark.parametrize(
    ("source_kind", "locator", "excerpt"),
    [
        (
            EvidenceSourceKind.LITERATURE_ABSTRACT,
            EvidenceLocator(paper_id="paper-1", doi="10.1/example"),
            "Abstract-level background statement.",
        ),
        (
            EvidenceSourceKind.USER_EXPERIMENT,
            EvidenceLocator(dataset_id="dataset-1"),
            None,
        ),
        (
            EvidenceSourceKind.DATA_ANALYSIS,
            EvidenceLocator(dataset_id="dataset-1", analysis_id="analysis-1"),
            None,
        ),
    ],
)
def test_source_roles_retain_distinct_structured_locators(
    source_kind: EvidenceSourceKind,
    locator: EvidenceLocator,
    excerpt: str | None,
) -> None:
    evidence = ScreeningEvidence(
        evidence_id=f"evidence-{source_kind.value}",
        project_id="project-1",
        source_kind=source_kind,
        source_id="source-1",
        locator=locator,
        source_excerpt=excerpt,
        review_status=EvidenceReviewStatus.VALIDATED,
    )

    assert evidence.source_kind is source_kind
    assert evidence.locator == locator


def test_source_kind_cannot_use_another_sources_locator() -> None:
    with pytest.raises(ValueError, match="query and material ids"):
        ScreeningEvidence(
            evidence_id="evidence-wrong-locator",
            project_id="project-1",
            source_kind=EvidenceSourceKind.MATERIALS_DATABASE,
            source_id="query-1",
            locator=EvidenceLocator(dataset_id="dataset-1"),
            review_status=EvidenceReviewStatus.VALIDATED,
        )
