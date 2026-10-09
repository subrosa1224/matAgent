"""Multiple exact-formula targets retain validated source snapshots."""

import json

import pytest

from materials_screening.models import MaterialRecord
from materials_screening.services.query_result_store import QueryResultStore
from materials_screening.sub_agents.materials_database.multi_query_handoff import (
    combine_verified_queries,
    explicit_multi_query_calls,
)

QUESTION = (
    "重点考察 In₂O₃、SnO₂ 和 ZnO。请先从 Materials Project 查询这三个体系中"
    "非金属、凸包能不高于 0.05 eV/atom 的结构，比较计算带隙；再检索实验文献。"
)


def test_explicit_multi_queries_use_valid_exact_formulas():
    from materials_screening.sub_agents.materials_database.models import (
        SearchMaterialsInput,
    )

    calls = explicit_multi_query_calls(QUESTION)
    assert calls is not None and len(calls) == 3
    args = [SearchMaterialsInput.model_validate_json(c.arguments) for c in calls]
    assert [a.formula for a in args] == ["In2O3", "SnO2", "ZnO"]
    assert all(a.chemsys is None for a in args)
    assert (
        explicit_multi_query_calls(QUESTION.replace("的结构", "的结构且密度不低于5"))
        is None
    )


def sources(tmp_path):
    store = QueryResultStore(tmp_path / "queries")
    for index, (formula, chemsys) in enumerate(
        (("In2O3", "In-O"), ("SnO2", "O-Sn"), ("ZnO", "O-Zn"))
    ):
        rows = [
            MaterialRecord(
                source="materials_project",
                material_id=f"mp-{index}",
                formula_pretty=formula,
                elements=chemsys.split("-"),
                chemsys=chemsys,
                is_metal=False,
                energy_above_hull_ev_atom=0.01,
                band_gap_ev=0.8,
            )
        ]
        if index == 1:
            rows.append(
                MaterialRecord(
                    source="materials_project",
                    material_id="mp-extra",
                    formula_pretty="SnO",
                    elements=("O", "Sn"),
                    chemsys=chemsys,
                    is_metal=False,
                    energy_above_hull_ev_atom=0.0,
                )
            )
        store.save_query(
            f"query-{index}",
            {
                "query_id": f"query-{index}",
                "source": "materials_project",
                "database_version": "test-v1",
                "request": {
                    "chemsys": chemsys,
                    "filters": [
                        {"field": "is_metal", "operator": "eq", "value": False},
                        {
                            "field": "energy_above_hull_ev_atom",
                            "operator": "lte",
                            "value": 0.05,
                        },
                    ],
                },
                "matched_count": len(rows),
                "stored_count": len(rows),
            },
            rows,
        )
    return store


def test_combines_only_requested_formulas_and_keeps_origin(tmp_path):
    store = sources(tmp_path)
    before = {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    query = combine_verified_queries(store, ("query-0", "query-1", "query-2"), QUESTION)
    rows = store.load_records(query)
    assert {r.formula_pretty for r in rows} == {"In2O3", "SnO2", "ZnO"}
    assert all(p.read_bytes() == content for p, content in before.items())
    metadata = json.loads((store.root / query / "metadata.json").read_text("utf-8"))
    assert metadata["source_query_ids"] == ["query-0", "query-1", "query-2"]
    assert metadata["excluded_material_ids"] == ["mp-extra"]
    assert (
        combine_verified_queries(store, ("query-0", "query-1", "query-2"), QUESTION)
        == query
    )


@pytest.mark.parametrize(
    "mutation", ["partial", "wrong_filter", "missing_source", "wrong_version"]
)
def test_ambiguous_or_incomplete_query_set_is_rejected(tmp_path, mutation):
    store = sources(tmp_path)
    ids = ("query-0", "query-1", "query-2")
    if mutation == "missing_source":
        ids = ids[:2]
    else:
        path = store.root / "query-1" / "metadata.json"
        data = json.loads(path.read_text("utf-8"))
        if mutation == "partial":
            data["matched_count"] += 1
        elif mutation == "wrong_filter":
            data["request"]["filters"][1]["value"] = 0.10
        else:
            data["database_version"] = "other"
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        combine_verified_queries(store, ids, QUESTION)


def test_current_turn_needs_input_final_is_replaced_by_verified_handoff(
    tmp_path, monkeypatch
):
    from materials_screening.agent.model_base import (
        AgentModelStatus,
        MaterialAgentResponse,
    )
    from materials_screening.agent.models import (
        AgentFunctionCallItem,
        AgentFunctionOutputItem,
        AgentMessageItem,
    )
    from materials_screening.sub_agents.materials_database import (
        multi_query_handoff as module,
    )
    from materials_screening.sub_agents.materials_database.models import (
        SearchMaterialsInput,
    )

    store = sources(tmp_path)
    monkeypatch.setattr(module, "QueryResultStore", lambda _root: store)
    items = []
    for index in range(3):
        meta = json.loads(
            (store.root / f"query-{index}" / "metadata.json").read_text("utf-8")
        )
        args = SearchMaterialsInput.model_validate(meta["request"])
        items.extend(
            (
                AgentFunctionCallItem(
                    call_id=f"c{index}",
                    name="search_materials",
                    arguments=args.model_dump_json(),
                ),
                AgentFunctionOutputItem(
                    call_id=f"c{index}",
                    output=json.dumps(
                        {
                            "status": "ok",
                            "tool_name": "search_materials",
                            "evidence_id": f"e{index}",
                            "output": {"query_id": f"query-{index}"},
                        }
                    ),
                ),
            )
        )
    response = MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        request_id="test",
        provider="test",
        model="test",
        output_items=(
            AgentMessageItem(
                role="assistant",
                content=json.dumps(
                    {"status": "needs_user_input", "answer": "请上传文献"}
                ),
            ),
        ),
    )
    actual = module.deliver_multi_query_handoff(response, items, QUESTION)
    assert actual is not None
    draft = json.loads(actual.message_text)
    assert (
        draft["status"] == "completed" and "MATERIAL_QUERY_HANDOFF" in draft["answer"]
    )
    assert draft["evidence_ids"] == ["e0", "e1", "e2"]
    assert "SnO |" not in draft["answer"]
    from materials_screening.agent.model_base import MaterialAgentRequest
    from materials_screening.sub_agents.materials_database.deterministic_model import (
        DeterministicDatabaseModel,
    )

    class NoDelegate:
        def generate(self, request):
            raise AssertionError(
                "Explicit verified query must not ask model to select papers"
            )

    request = MaterialAgentRequest(
        input_items=(AgentMessageItem(role="user", content=QUESTION), *items),
        instructions="database",
        tool_definitions=(),
        final_draft_schema={"type": "object"},
    )
    routed = DeterministicDatabaseModel(NoDelegate()).generate(request)
    assert "MATERIAL_QUERY_HANDOFF" in routed.message_text
    failed_items = list(items)
    failed_items[-1] = AgentFunctionOutputItem(
        call_id="c2", output='{"status":"error"}'
    )
    from dataclasses import replace

    failed_request = replace(
        request,
        input_items=(AgentMessageItem(role="user", content=QUESTION), *failed_items),
    )
    failed = DeterministicDatabaseModel(NoDelegate()).generate(failed_request)
    assert json.loads(failed.message_text)["status"] == "error"
    assert "MATERIAL_QUERY_HANDOFF" not in failed.message_text


def test_unparsed_extra_condition_is_not_silently_discarded(tmp_path):
    store = sources(tmp_path)
    with pytest.raises(ValueError):
        combine_verified_queries(
            store,
            ("query-0", "query-1", "query-2"),
            QUESTION.replace("比较计算带隙", "带隙不低于 2 eV，比较计算带隙"),
        )
