"""A bounded full-chain database task must finish from verified local evidence."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.models import MaterialRecord
from materials_screening.services.material_database_service import project_record
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.materials_database.deterministic_model import (
    DeterministicDatabaseModel,
)
from materials_screening.sub_agents.materials_database.models import (
    SearchMaterialsInput,
)
from materials_screening.sub_agents.materials_database.screening_handoff import (
    SCREENING_HANDOFF_SCOPE,
    scoped_screening_handoff,
    with_screening_handoff_scope,
)

QUESTION = (
    "请从 Materials Project 筛选化学体系严格为 Li-Fe-O、"
    "凸包能不高于 0.10 eV/atom 的非金属材料，"
    "分析不同化学式和晶系的稳定性、计算带隙及密度分布，"
    "并检索正极实验文献，评估应用证据。"
)


def request(message, *items):
    return MaterialAgentRequest(
        instructions="database",
        input_items=(AgentMessageItem(role="user", content=message), *items),
        tool_definitions=(),
        final_draft_schema={"type": "object"},
    )


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = SearchMaterialsInput(
        chemsys="Li-Fe-O",
        filters=(
            {"field": "energy_above_hull_ev_atom", "operator": "lte", "value": 0.1},
            {"field": "is_metal", "operator": "eq", "value": False},
        ),
        fields=(
            "material_id",
            "formula_pretty",
            "band_gap_ev",
            "density_g_cm3",
            "energy_above_hull_ev_atom",
            "is_metal",
            "elements",
            "crystal_system",
        ),
        limit=100,
    )
    row = MaterialRecord(
        source="materials_project",
        material_id="mp-1",
        formula_pretty="LiFeO2",
        elements=("Li", "Fe", "O"),
        chemsys="Fe-Li-O",
        band_gap_ev=1.2,
        density_g_cm3=3.4,
        energy_above_hull_ev_atom=0.08,
        is_metal=False,
        symmetry={"crystal_system": "Cubic", "number": 225, "symbol": "Fm-3m"},
    )
    metadata = {
        "query_id": "query-scope",
        "source": "materials_project",
        "request": args.model_dump(mode="json"),
        "matched_count": 1,
        "stored_count": 1,
    }
    store = QueryResultStore()
    store.save_query("query-scope", metadata, (row,))
    payload = {
        "query_id": "query-scope",
        "source": "materials_project",
        "matched_count": 1,
        "returned_count": 1,
        "fields": list(args.fields),
        "warnings": [],
        "materials": [project_record(row, args.fields)],
    }
    call = AgentFunctionCallItem(
        call_id="search", name="search_materials", arguments=args.model_dump_json()
    )
    output = AgentFunctionOutputItem(
        call_id="search",
        output=json.dumps(
            {
                "status": "ok",
                "tool_name": "search_materials",
                "evidence_id": "e-search",
                "output": payload,
            }
        ),
    )
    return args, call, output, metadata


def test_scope_finishes_without_another_model_call(snapshot):
    _, call, output, _ = snapshot
    delegate = Mock()
    result = DeterministicDatabaseModel(delegate).generate(
        request(QUESTION + SCREENING_HANDOFF_SCOPE, call, output)
    )
    delegate.generate.assert_not_called()
    draft = json.loads(result.message_text)
    assert draft["status"] == "completed"
    assert "MATERIAL_QUERY_HANDOFF: query_id=query-scope" in draft["answer"]
    assert "性能最佳" not in draft["answer"]


@pytest.mark.parametrize(
    "change",
    [
        "no_scope",
        "extra_filter",
        "wrong_chemsys",
        "wrong_hull",
        "multiple_queries",
        "empty",
        "failed",
        "wrong_metadata",
        "wrong_record",
        "missing_snapshot",
        "metadata_count",
        "unpaired_output",
        "details",
        "export",
        "additional_condition",
        "missing_field",
        "previous_turn",
        "duplicate_output",
        "invalid_query_id",
    ],
)
def test_unsafe_or_other_work_cannot_be_early_finished(snapshot, change):
    args, call, output, metadata = snapshot
    message = QUESTION + SCREENING_HANDOFF_SCOPE
    items = [call, output]
    envelope = json.loads(output.output)
    if change == "no_scope":
        message = QUESTION
    elif change in {"extra_filter", "wrong_chemsys", "wrong_hull", "missing_field"}:
        data = args.model_dump(mode="json")
        if change == "extra_filter":
            data["filters"].append(
                {"field": "density_g_cm3", "operator": "gte", "value": 4}
            )
        elif change == "wrong_chemsys":
            data["chemsys"] = "Na-Fe-O"
        elif change == "wrong_hull":
            data["filters"][0]["value"] = 0.2
        else:
            data["fields"].remove("density_g_cm3")
        items[0] = call.model_copy(update={"arguments": json.dumps(data)})
    elif change == "multiple_queries":
        second = json.loads(output.output)
        second["output"]["query_id"] = "query-other"
        items += [
            call.model_copy(update={"call_id": "other"}),
            output.model_copy(
                update={"call_id": "other", "output": json.dumps(second)}
            ),
        ]
    elif change in {"empty", "failed", "invalid_query_id"}:
        if change == "empty":
            envelope["output"].update(matched_count=0, returned_count=0, materials=[])
        elif change == "failed":
            envelope["status"] = "error"
        else:
            envelope["output"]["query_id"] = "../query-scope"
        items[1] = output.model_copy(update={"output": json.dumps(envelope)})
    elif change in {"wrong_metadata", "metadata_count"}:
        if change == "wrong_metadata":
            metadata["request"]["chemsys"] = "Na-Fe-O"
        else:
            metadata["stored_count"] = 2
        Path("data/material_queries/query-scope/metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    elif change == "wrong_record":
        path = Path("data/material_queries/query-scope/records.jsonl")
        row = json.loads(path.read_text(encoding="utf-8"))
        row["energy_above_hull_ev_atom"] = 0.2
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif change == "missing_snapshot":
        Path("data/material_queries/query-scope/records.jsonl").unlink()
    elif change == "unpaired_output":
        items = [output]
    elif change in {"details", "export"}:
        message = (
            QUESTION
            + ("另请查看详情。" if change == "details" else "另请导出CSV。")
            + SCREENING_HANDOFF_SCOPE
        )
    elif change == "additional_condition":
        message = QUESTION + "还要求密度大于4 g/cm3。" + SCREENING_HANDOFF_SCOPE
    elif change == "previous_turn":
        items += [AgentMessageItem(role="user", content=message)]
    elif change == "duplicate_output":
        items.append(output)
    assert scoped_screening_handoff(request(message, *items)) is None


def test_initial_schema_error_does_not_hide_later_verified_query(snapshot):
    _, call, output, _ = snapshot
    invalid = AgentFunctionCallItem(
        call_id="bad", name="search_materials", arguments='{"fields":"bad"}'
    )
    failed = AgentFunctionOutputItem(
        call_id="bad",
        output=json.dumps(
            {
                "status": "error",
                "tool_name": "search_materials",
                "error": {"code": "INVALID_ARGUMENTS"},
            }
        ),
    )
    result = scoped_screening_handoff(
        request(QUESTION + SCREENING_HANDOFF_SCOPE, invalid, failed, call, output)
    )
    assert result is not None


def test_other_strict_chemical_system_uses_its_own_constraints(snapshot):
    assert (
        scoped_screening_handoff(
            request(
                QUESTION.replace("Li-Fe-O", "Na-Mn-O") + SCREENING_HANDOFF_SCOPE,
                snapshot[1],
                snapshot[2],
            )
        )
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("band_gap_ev", 99),
        ("density_g_cm3", 99),
        ("formula_pretty", "NaMnO2"),
        ("material_id", "mp-fake"),
        ("is_metal", True),
    ],
)
def test_display_page_must_exactly_match_stored_records(snapshot, field, value):
    _, call, output, _ = snapshot
    envelope = json.loads(output.output)
    envelope["output"]["materials"][0][field] = value
    output = output.model_copy(update={"output": json.dumps(envelope)})
    assert (
        scoped_screening_handoff(
            request(QUESTION + SCREENING_HANDOFF_SCOPE, call, output)
        )
        is None
    )


@pytest.mark.parametrize(
    "suffix",
    [
        "另要求只要理论材料。",
        "另外排除Li。",
        "仅限密度在2～4之间。",
        "另请排序。",
    ],
)
def test_unrecognized_extra_conditions_do_not_receive_scope(suffix):
    message = QUESTION + suffix
    assert with_screening_handoff_scope(message) == message


def test_scope_applies_to_generic_explicit_filter_not_a_question_id():
    message = QUESTION.replace("Li-Fe-O", "Na-Mn-O").replace("0.10", "0.07")
    assert with_screening_handoff_scope(message) == message + SCREENING_HANDOFF_SCOPE


def test_verified_other_chemical_system_can_finish(snapshot):
    args, _, output, metadata = snapshot
    data = args.model_dump(mode="json")
    data["chemsys"] = "Na-Mn-O"
    data["filters"][0]["value"] = 0.07
    args = SearchMaterialsInput.model_validate(data)
    path = Path("data/material_queries/query-scope/records.jsonl")
    row = json.loads(path.read_text(encoding="utf-8"))
    row.update(
        elements=["Na", "Mn", "O"],
        chemsys="Mn-Na-O",
        formula_pretty="NaMnO2",
        energy_above_hull_ev_atom=0.04,
    )
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    metadata["request"] = args.model_dump(mode="json")
    Path("data/material_queries/query-scope/metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    envelope = json.loads(output.output)
    envelope["output"]["materials"] = [
        project_record(MaterialRecord.model_validate(row), args.fields)
    ]
    call = AgentFunctionCallItem(
        call_id="search", name="search_materials", arguments=args.model_dump_json()
    )
    output = output.model_copy(update={"output": json.dumps(envelope)})
    message = QUESTION.replace("Li-Fe-O", "Na-Mn-O").replace("0.10", "0.07")
    assert (
        scoped_screening_handoff(
            request(message + SCREENING_HANDOFF_SCOPE, call, output)
        )
        is not None
    )
