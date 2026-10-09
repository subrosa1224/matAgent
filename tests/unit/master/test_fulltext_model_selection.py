"""Shared service/model selection must agree with saved extraction profiles."""

import pytest

from materials_screening.master.artifact_registry import ArtifactRegistry
from materials_screening.planner.settings import Settings
from materials_screening.sub_agents.literature import fulltext_factory as factory


@pytest.mark.parametrize(
    "override, expected",
    [(None, "configured-main"), ("custom-extractor", "custom-extractor")],
)
def test_fulltext_model_and_snapshot_profile_share_selection(
    tmp_path, monkeypatch, override, expected
):
    monkeypatch.setattr(
        factory,
        "Settings",
        lambda: Settings(_env_file=None, intern_model="configured-main"),
    )
    if override is None:
        monkeypatch.delenv("LITERATURE_EXTRACTION_MODEL", raising=False)
    else:
        monkeypatch.setenv("LITERATURE_EXTRACTION_MODEL", override)
    captured = []
    monkeypatch.setattr(
        factory, "create_llm_provider", lambda settings: captured.append(settings)
    )
    factory._ConfiguredLlm()
    assert captured[0].intern_model == expected
    processor = factory.create_fulltext_preview_processor(
        ArtifactRegistry(tmp_path), enable_remote=True
    )
    assert processor.extraction_processor.model_profile == "intern/" + expected
    assert not processor.extraction_processor.source_selection


@pytest.mark.parametrize(
    "override,expected",
    [(None, "configured-main"), ("custom-extractor", "custom-extractor")],
)
def test_staged_path_opt_in_uses_main_model_but_respects_explicit_override(
    tmp_path, monkeypatch, override, expected
):
    from materials_screening.master.staged_fulltext_processor import (
        StagedFulltextProcessor,
    )

    monkeypatch.setattr(
        factory,
        "Settings",
        lambda: Settings(_env_file=None, intern_model="configured-main"),
    )
    monkeypatch.setenv("MASTER_STAGED_FULLTEXT_EXTRACTION", "true")
    if override is None:
        monkeypatch.delenv("LITERATURE_EXTRACTION_MODEL", raising=False)
    else:
        monkeypatch.setenv("LITERATURE_EXTRACTION_MODEL", override)
    captured = []
    monkeypatch.setattr(
        factory, "create_llm_provider", lambda settings: captured.append(settings)
    )
    processor = factory.create_fulltext_preview_processor(
        ArtifactRegistry(tmp_path), enable_remote=True
    )
    staged = processor.extraction_processor
    assert isinstance(staged, StagedFulltextProcessor)
    assert staged.model_profile == "intern/" + expected
    staged.llm_factory()
    assert captured[0].intern_model == expected


def test_fulltext_explicit_dotenv_model_is_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LITERATURE_EXTRACTION_MODEL", raising=False)
    monkeypatch.delenv("INTERN_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "INTERN_MODEL=lab-main\nLITERATURE_EXTRACTION_MODEL=lab-extractor\n",
        encoding="utf8",
    )
    captured = []
    monkeypatch.setattr(
        factory, "create_llm_provider", lambda settings: captured.append(settings)
    )
    factory._ConfiguredLlm()
    assert captured[0].intern_model == "lab-extractor"
