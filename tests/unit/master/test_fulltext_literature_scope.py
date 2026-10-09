"""Stage-scoped preview repair with offline sources, not scientific acceptance."""

from materials_screening.master.fulltext_preview import (
    _literature_boundary_text,
)
from materials_screening.sub_agents.literature.preview import (
    BoundaryFinding,
    PaperPreviewExtractor,
    PreviewBoundaryPolicy,
)
from tests.unit.master.test_fulltext_batches import batch
from tests.unit.master.test_fulltext_preview import run, setup

QUESTION = (
    "我想寻找透明电极候选材料，重点考察 In₂O₃、SnO₂ 和 ZnO。"
    "请先从 Materials Project 查询非金属、凸包能不高于 0.05 eV/atom 的结构，"
    "比较计算带隙；再检索这些材料及其掺杂体系作为透明导电薄膜的实验研究。"
    "上传全文并确认后，提取透过率、电阻率及制备条件。"
    "区分未掺杂结构的计算属性与掺杂薄膜的实验性能；条件不可比时不要排名。"
)


def scoped(env):
    env.task = env.task.model_copy(update={"original_question": QUESTION})
    return env


def corrupt_scope(task):
    """Emulate the saved scope-enforcement defect without corrupting quotes."""
    previews = []
    for p in task.previews:
        finding = BoundaryFinding(
            finding_type="material_scope",
            severity="exclude",
            reason="掺杂薄膜不符合数据库未掺杂结构要求，且没有同时研究三个体系。",
            evidence_quote=p.evidence_quote,
            chunk_id=p.chunk_id,
        )
        previews.append(
            p.model_copy(
                update={
                    "boundary_findings": (finding,),
                    "topic_relevance": "low",
                    "recommendation": "exclude",
                }
            )
        )
    return task.model_copy(update={"previews": tuple(previews)})


def test_scoped_model_input_keeps_paper_question_not_database_conditions(tmp_path):
    env = scoped(setup(tmp_path))
    requests = []
    generate = env.llm.generate_structured

    def record(**kwargs):
        requests.append(kwargs)
        return generate(**kwargs)

    env.llm.generate_structured = record
    run(env)
    topic = requests[0]["user_text"].split("Paper title:", 1)[0]
    assert "透明电极" in topic
    assert "SnO₂" in topic
    assert "掺杂体系" in topic
    assert "Materials Project" not in topic
    assert "0.05" not in topic
    assert "区分未掺杂" not in topic
    assert env.saved[-1].previews[0].topic == QUESTION


def test_explicit_empty_paper_policy_rejects_model_invented_hard_boundary(tmp_path):
    env = setup(tmp_path)
    generate = env.llm.generate_structured

    def invented_boundary(**kwargs):
        response = generate(**kwargs)
        candidate = response.parsed
        finding = BoundaryFinding(
            finding_type="material_scope",
            severity="exclude",
            reason="没有同时涉及三个材料体系。",
            evidence_quote=candidate.evidence_quote,
            chunk_id=candidate.chunk_id,
        )
        return response.model_copy(
            update={
                "parsed": candidate.model_copy(update={"boundary_findings": (finding,)})
            }
        )

    env.llm.generate_structured = invented_boundary
    run(env)
    assert env.saved[-1].previews[0].boundary_findings == ()
    assert env.saved[-1].previews[0].recommendation == "deep_analyze"


def test_extractor_scoped_topic_preserves_original_binding(tmp_path):
    env = scoped(setup(tmp_path))
    scoped_topic = _literature_boundary_text(QUESTION)
    p = PaperPreviewExtractor(env.llm, boundary_policy=PreviewBoundaryPolicy()).extract(
        document_id=env.task.document_ids[0],
        title="Paper",
        topic=QUESTION,
        screening_topic=scoped_topic,
        chunks=env.store.chunks,
    )
    assert p.topic == QUESTION


def test_cached_bad_scope_is_rechecked_once_with_history_then_fresh_consent(tmp_path):
    env = scoped(setup(tmp_path, 2))
    batch(env)
    old = corrupt_scope(env.saved[-1]).model_copy(
        update={
            "user_instructions": ("确认详细分析全部内容",),
        }
    )
    result = run(env, old)
    repaired = env.saved[-1]
    assert result.final_status == "needs_user_input"
    assert "确认更新后的预览" in result.response_text
    assert "当前仅有背景阅读" not in result.response_text
    assert env.llm.calls == 4
    assert len(repaired.preview_retry_history) == 2
    assert all(a.status == "succeeded" for a in repaired.preview_retry_history)
    assert all(
        a.previous_preview in old.previews for a in repaired.preview_retry_history
    )
    assert all(p.recommendation == "deep_analyze" for p in repaired.previews)
    assert all(d.action == "hold" for d in repaired.preview_decisions.values())
    assert all("重新核验" in d.reason for d in repaired.preview_decisions.values())
    assert not repaired.extraction_snapshots
    assert repaired.original_question == QUESTION
    assert repaired.artifact_refs == old.artifact_refs
    confirmed = repaired.model_copy(
        update={
            "user_instructions": (*repaired.user_instructions, "确认详细分析全部内容"),
        }
    )
    result = run(env, confirmed)
    assert result.error["code"] == "FULLTEXT_EXTRACTION_UNAVAILABLE"
    assert env.llm.calls == 4
    assert all(d.action == "extract" for d in env.saved[-1].preview_decisions.values())


def test_cached_bad_scope_cannot_hide_a_forged_quote(tmp_path):
    env = scoped(setup(tmp_path))
    run(env)
    old = corrupt_scope(env.saved[-1])
    forged = old.model_copy(
        update={
            "previews": (
                old.previews[0].model_copy(update={"evidence_quote": "forged text"}),
            )
        }
    )
    result = run(env, forged)
    assert result.error["code"] == "FULLTEXT_PREVIEW_FAILED"
    assert env.llm.calls == 1
    assert not env.saved[-1].preview_retry_history


def test_scope_recheck_provider_failure_keeps_old_preview_without_auto_retry(tmp_path):
    env = scoped(setup(tmp_path))
    batch(env)
    old = corrupt_scope(env.saved[-1])
    env.llm.fail = True
    result, _ = batch(type(env)(**{**vars(env), "task": old}))
    assert result.error["code"] == "FULLTEXT_PREVIEW_FAILED"
    assert env.llm.calls == 2
    assert env.saved[-1].previews == old.previews
    assert env.saved[-1].preview_retry_history[-1].status == "failed"


def test_ten_cached_scope_errors_auto_recheck_in_bounded_batches(tmp_path):
    env = scoped(setup(tmp_path, 10))
    batch(env)
    env.task = corrupt_scope(env.saved[-1])
    result, _ = batch(env, batch_model_budget=3)
    repaired = env.saved[-1]
    assert result.final_status == "needs_user_input"
    assert result.model_call_count == 10
    assert env.llm.calls == 20
    assert len(repaired.preview_retry_history) == 10
    assert all(a.status == "succeeded" for a in repaired.preview_retry_history)
    assert all(d.action == "hold" for d in repaired.preview_decisions.values())
    assert not repaired.extraction_snapshots


def test_scope_recheck_never_exceeds_existing_two_attempt_limit(tmp_path):
    env = scoped(setup(tmp_path))
    batch(env)
    env.task = corrupt_scope(env.saved[-1])
    env.llm.fail = True
    for _ in range(3):
        batch(env)
        env.task = env.saved[-1]
    assert env.llm.calls == 3  # One original preview plus two failed scope checks.
    assert len(env.task.preview_retry_history) == 2
    assert not env.task.extraction_snapshots


def test_explicit_undoped_paper_constraint_is_not_relaxed(tmp_path):
    env = setup(tmp_path)
    question = "查询稳定结构；检索仅限未掺杂 ZnO 的实验论文，提取性能。"
    env.task = env.task.model_copy(update={"original_question": question})
    run(env)
    old = corrupt_scope(env.saved[-1])
    run(env, old)
    assert env.llm.calls == 1
    assert not env.saved[-1].preview_retry_history
    assert env.saved[-1].preview_decisions[old.document_ids[0]].action == "exclude"
