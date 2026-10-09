"""A master-owned explicit formula query must bind all filters and its snapshot."""

import json
import re
from dataclasses import replace
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

SUITE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures/research/stable_acceptance_v1/suite.json"
    ).read_text(encoding="utf-8")
)
CASES = [next(c for c in SUITE["cases"] if c["id"] == x) for x in ("N02", "N04", "N08")]
FIELDS = (
    "material_id",
    "formula_pretty",
    "elements",
    "crystal_system",
    "band_gap_ev",
    "density_g_cm3",
    "energy_above_hull_ev_atom",
    "is_metal",
    "is_stable",
    "formation_energy_ev_atom",
)


def request(message, *items):
    return MaterialAgentRequest(
        instructions="database",
        input_items=(AgentMessageItem(role="user", content=message), *items),
        tool_definitions=(),
        final_draft_schema={"type": "object"},
    )


def expected(case):
    constraints = case["constraints"]
    filters = [
        {"field": "is_metal", "operator": "eq", "value": False},
        {
            "field": "energy_above_hull_ev_atom",
            "operator": "lte",
            "value": constraints["energy_above_hull_ev_atom"][1],
        },
    ]
    if constraints.get("crystal_system"):
        filters.append(
            {
                "field": "crystal_system",
                "operator": "eq",
                "value": constraints["crystal_system"],
            }
        )
    return SearchMaterialsInput(
        formula=constraints["formulas"][0], filters=filters, fields=FIELDS, limit=100
    )


@pytest.mark.parametrize("case", CASES, ids=("N02", "N04", "N08"))
def test_original_query_receives_snapshot_only_scope(case):
    question = case["question"]
    assert with_screening_handoff_scope(question) == question + SCREENING_HANDOFF_SCOPE


@pytest.mark.parametrize("case", CASES, ids=("N02", "N04", "N08"))
def test_first_call_includes_every_explicit_constraint(case):
    delegate = Mock(side_effect=AssertionError("No unbound model call permitted"))
    delegate.generate = Mock(
        side_effect=AssertionError("No unbound model call permitted")
    )
    result = DeterministicDatabaseModel(delegate).generate(
        request(case["question"] + SCREENING_HANDOFF_SCOPE)
    )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_materials"
    actual = SearchMaterialsInput.model_validate_json(result.tool_calls[0].arguments)
    assert actual == expected(case)
    delegate.generate.assert_not_called()


@pytest.fixture(params=CASES, ids=("N02", "N04", "N08"))
def snapshot(request, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = request.param
    args = expected(case)
    formula = args.formula
    row = MaterialRecord(
        source="materials_project",
        material_id="mp-formula",
        formula_pretty=formula,
        elements=tuple(re.findall(r"[A-Z][a-z]?", formula)),
        is_metal=False,
        energy_above_hull_ev_atom=0.01,
        band_gap_ev=1.0,
        density_g_cm3=4.0,
        symmetry={"crystal_system": case["constraints"].get("crystal_system", "Cubic")},
    )
    metadata = {
        "query_id": "query-formula",
        "source": "materials_project",
        "request": args.model_dump(mode="json"),
        "matched_count": 1,
        "stored_count": 1,
    }
    QueryResultStore().save_query("query-formula", metadata, (row,))
    call = AgentFunctionCallItem(
        call_id="formula-search",
        name="search_materials",
        arguments=args.model_dump_json(),
    )
    output = AgentFunctionOutputItem(
        call_id=call.call_id,
        output=json.dumps(
            {
                "status": "ok",
                "tool_name": "search_materials",
                "evidence_id": "formula-evidence",
                "output": {
                    "query_id": "query-formula",
                    "source": "materials_project",
                    "matched_count": 1,
                    "returned_count": 1,
                    "fields": list(args.fields),
                    "materials": [project_record(row, args.fields)],
                },
            }
        ),
    )
    return case, args, row, metadata, call, output


def test_only_verified_formula_snapshot_finishes(snapshot):
    case, _, _, _, call, output = snapshot
    delegate = Mock()
    result = DeterministicDatabaseModel(delegate).generate(
        request(case["question"] + SCREENING_HANDOFF_SCOPE, call, output)
    )
    delegate.generate.assert_not_called()
    draft = json.loads(result.message_text)
    assert draft["status"] == "completed"
    assert "MATERIAL_QUERY_HANDOFF: query_id=query-formula" in draft["answer"]


@pytest.mark.parametrize(
    "damage",
    (
        "omitted_filter",
        "wrong_formula",
        "wrong_hull",
        "wrong_row",
        "bad_metadata",
        "wrong_page",
        "missing_snapshot",
    ),
)
def test_bad_or_incomplete_snapshot_cannot_be_passed_to_analysis(snapshot, damage):
    case, args, _, metadata, call, output = snapshot
    if damage in {"omitted_filter", "wrong_formula", "wrong_hull"}:
        raw = args.model_dump(mode="json")
        if damage == "omitted_filter":
            raw["filters"] = raw["filters"][1:]
        elif damage == "wrong_formula":
            raw["formula"] = "TiO2"
        else:
            raw["filters"][1]["value"] = 0.9
        call = call.model_copy(update={"arguments": json.dumps(raw)})
    elif damage == "wrong_row":
        path = Path("data/material_queries/query-formula/records.jsonl")
        row = json.loads(path.read_text(encoding="utf-8"))
        row["is_metal"] = True
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif damage == "bad_metadata":
        metadata["request"]["filters"] = []
        Path("data/material_queries/query-formula/metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    elif damage == "wrong_page":
        raw = json.loads(output.output)
        raw["output"]["materials"][0]["density_g_cm3"] = 99
        output = output.model_copy(update={"output": json.dumps(raw)})
    else:
        Path("data/material_queries/query-formula/records.jsonl").unlink()
    value = request(case["question"] + SCREENING_HANDOFF_SCOPE, call, output)
    assert scoped_screening_handoff(value) is None
    delegate = Mock()
    result = DeterministicDatabaseModel(delegate).generate(value)
    delegate.generate.assert_not_called()
    draft = json.loads(result.message_text)
    assert draft["status"] == "error"
    assert "MATERIAL_QUERY_HANDOFF" not in draft["answer"]


def test_other_formula_is_not_a_case_specific_rule():
    question = (
        "请从 Materials Project 查询化学式为NaMnO2、"
        "非金属且凸包能不高于0.07 eV/atom的候选，分析密度；检索电极实验论文。"
    )
    assert with_screening_handoff_scope(question).endswith(SCREENING_HANDOFF_SCOPE)
    delegate = Mock()
    result = DeterministicDatabaseModel(delegate).generate(
        request(question + SCREENING_HANDOFF_SCOPE)
    )
    args = SearchMaterialsInput.model_validate_json(result.tool_calls[0].arguments)
    assert args.formula == "NaMnO2"
    assert args.filters[1].value == 0.07


@pytest.mark.parametrize(
    "suffix",
    ("另外要求密度大于4。", "另请导出CSV。", "另外排除Fe。", "另要求只要理论材料。"),
)
def test_extra_conditions_cannot_be_silently_dropped(suffix):
    question = CASES[2]["question"] + suffix
    assert with_screening_handoff_scope(question) == question


def test_standalone_query_is_not_forced_into_master_snapshot_mode():
    delegate = Mock()
    DeterministicDatabaseModel(delegate).generate(request(CASES[2]["question"]))
    delegate.generate.assert_called_once()


@pytest.mark.parametrize("damage", ("hull", "elements", "crystal"))
def test_every_stored_row_is_checked_not_only_display_page(snapshot, damage):
    case, args, row, metadata, call, output = snapshot
    if damage == "crystal" and case["id"] != "N08":
        pytest.skip("Only N08 requests a specific crystal system")
    second = row.model_dump(mode="json")
    second["material_id"] = "mp-hidden"
    if damage == "hull":
        second["energy_above_hull_ev_atom"] = 0.9
    elif damage == "elements":
        second["elements"] = ["Pb", "O"]
    else:
        second["symmetry"]["crystal_system"] = "Cubic"
    args = args.model_copy(update={"limit": 1})
    metadata.update(
        query_id="query-hidden",
        request=args.model_dump(mode="json"),
        matched_count=2,
        stored_count=2,
    )
    QueryResultStore().save_query(
        "query-hidden", metadata, (row, MaterialRecord.model_validate(second))
    )
    call = call.model_copy(update={"arguments": args.model_dump_json()})
    raw = json.loads(output.output)
    raw["output"]["matched_count"] = 2
    raw["output"]["query_id"] = "query-hidden"
    output = output.model_copy(update={"output": json.dumps(raw)})
    value = request(case["question"] + SCREENING_HANDOFF_SCOPE, call, output)
    assert scoped_screening_handoff(value) is None
    draft = json.loads(DeterministicDatabaseModel(Mock()).generate(value).message_text)
    assert draft["status"] == "error"


def test_no_tool_calls_when_existing_budget_disallows_them():
    delegate = Mock()
    value = request(CASES[2]["question"] + SCREENING_HANDOFF_SCOPE)
    value = replace(value, allow_tool_calls=False)
    result = DeterministicDatabaseModel(delegate).generate(value)
    assert not result.tool_calls
    assert json.loads(result.message_text)["status"] == "error"
    delegate.generate.assert_not_called()
