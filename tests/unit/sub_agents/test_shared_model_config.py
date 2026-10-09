"""All checks use fake models; no connection or credentials are sent."""

import importlib
from types import SimpleNamespace

import pytest

from materials_screening.agent_tools.result_reader import FileWorkflowResultReader


@pytest.mark.parametrize(
    "name",
    [
        "literature",
        "materials_database",
        "data_analysis",
        "materials_screening",
        "outlier_detection",
    ],
)
@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_subagent_uses_shared_model_config(tmp_path, monkeypatch, name, source):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INTERN_BASE_URL", "http://lab.invalid:23333/v1/")
    monkeypatch.setenv("INTERN_MODEL", "lab-model")
    monkeypatch.setenv("INTERN_THINKING_MODE", "false")
    if source == "dotenv":
        for key in ("INTERN_BASE_URL", "INTERN_MODEL", "INTERN_THINKING_MODE"):
            monkeypatch.delenv(key, raising=False)
        (tmp_path / ".env").write_text(
            "INTERN_BASE_URL=http://lab.invalid:23333/v1/\n"
            "INTERN_MODEL=lab-model\nINTERN_THINKING_MODE=false\n",
            encoding="utf8",
        )
    monkeypatch.setenv("AGENT_BASE_URL", "https://old.invalid/v1/")
    monkeypatch.setenv("AGENT_MODEL", "old-model")
    module = importlib.import_module(
        f"materials_screening.sub_agents.{name}.spec_factory"
    )
    captured = []

    def model(settings, **kwargs):
        captured.append(settings)
        return SimpleNamespace()

    monkeypatch.setattr(
        "materials_screening.agent.intern_model.InternAgentModel", model
    )
    if hasattr(module, "InternAgentModel"):
        monkeypatch.setattr(module, "InternAgentModel", model)
    monkeypatch.setattr(
        module, "MaterialAgentRunner", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    args = dict(workflow_runner=object(), intern_api_key="not-a-real-key")
    if name != "materials_screening":
        args["result_reader"] = FileWorkflowResultReader(tmp_path / "runs")
    if name in {"materials_database", "outlier_detection"}:
        args["repository"] = object()
    if name == "literature":
        args.update(enable_rag=False, enable_remote_search=False)
    module.create_spec(**args).runner_factory()
    assert len(captured) == 1
    assert captured[0].agent_base_url.rstrip("/") == "http://lab.invalid:23333/v1"
    assert captured[0].agent_model == "lab-model"
    assert captured[0].agent_thinking_mode is False


@pytest.mark.parametrize("builder", ["_build_agent_model", "_build_master_model"])
def test_cli_main_model_uses_shared_config(tmp_path, monkeypatch, builder):
    from materials_screening import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INTERN_API_KEY", "not-a-real-key")
    monkeypatch.setenv("INTERN_BASE_URL", "http://lab.invalid:23333/v1/")
    monkeypatch.setenv("INTERN_MODEL", "lab-model")
    monkeypatch.setenv("INTERN_THINKING_MODE", "false")
    captured = []

    def model(settings, **kwargs):
        captured.append(settings)
        return SimpleNamespace()

    monkeypatch.setattr(
        "materials_screening.agent.intern_model.InternAgentModel", model
    )
    getattr(cli, builder)("intern")
    assert captured[0].agent_base_url == "http://lab.invalid:23333/v1"
    assert captured[0].agent_model == "lab-model"
    assert captured[0].agent_thinking_mode is False


def test_shared_mapping_retains_agent_limits_and_explicit_ui_thinking(
    tmp_path, monkeypatch
):
    from materials_screening.agent.settings import AgentSettings, shared_intern_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INTERN_MODEL", "lab-model")
    monkeypatch.setenv("INTERN_THINKING_MODE", "true")
    original = AgentSettings(
        agent_max_model_calls_per_turn=6, agent_system_prompt="custom"
    )
    configured = shared_intern_settings(original, thinking_mode=False)
    assert configured.agent_thinking_mode is False
    assert configured.agent_max_model_calls_per_turn == 6
    assert configured.agent_system_prompt == "custom"
    assert original.agent_thinking_mode is True
