import pytest
from tests.unit.sub_agents.test_literature_matrix_automation import (
    FakeLlm,
    Store,
    _chunk,
)

from materials_screening.llm.errors import LLMStructuredOutputError, LLMTimeoutError
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
    MatrixExtractionBatch,
)

SECRET = "secret-must-not-appear"


class FailingLlm(FakeLlm):
    def __init__(self, error: Exception, recover: bool = False) -> None:
        super().__init__(MatrixExtractionBatch())
        self.error = error
        self.recover = recover
        self.calls = 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        if not self.recover or self.calls == 1:
            raise self.error
        return super().generate_structured(**kwargs)


@pytest.mark.parametrize(
    "error,category,location",
    [
        (
            LLMStructuredOutputError("Intern response exceeded max_tokens"),
            "output_truncated",
            None,
        ),
        (
            LLMStructuredOutputError("Intern returned empty structured output"),
            "empty_output",
            None,
        ),
        (
            LLMStructuredOutputError("Intern output is not valid JSON"),
            "invalid_json",
            None,
        ),
        (
            LLMStructuredOutputError(
                "Intern output is not valid JSON "
                "(provider reported prompt processing error)"
            ),
            "prompt_processing_error",
            None,
        ),
        (
            LLMStructuredOutputError(
                "Intern output does not match schema at measurements.0.group_key: "
                f"{SECRET}"
            ),
            "schema_mismatch",
            "measurements.0.group_key",
        ),
        (
            LLMStructuredOutputError(
                "Intern output does not match schema at "
                f"groups.0.variables.{SECRET}: invalid"
            ),
            "schema_mismatch",
            None,
        ),
        (LLMStructuredOutputError(SECRET), "structured_output_error", None),
        (LLMTimeoutError(SECRET), "timeout", None),
    ],
)
def test_batch_failures_keep_safe_specific_cause(error, category, location) -> None:
    llm = FailingLlm(error)
    result = AutomatedMatrixExtractor(llm, Store()).extract(
        document_id="doc-1",
        chunks=(_chunk(),),
        max_output_tokens=1234,
    )
    attempts = result.diagnostics.batch_attempts
    assert llm.calls == 2
    assert len(attempts) == 2
    assert [a.attempt for a in attempts] == [1, 2]
    assert all(a.batch_index == 1 and a.status == "error" for a in attempts)
    assert all(
        a.error_category == category and a.schema_location == location for a in attempts
    )
    assert all(
        a.chunk_ids == ("chunk-1",) and a.page_ranges == ((2, 2),) for a in attempts
    )
    assert all(a.output_token_budget == 1234 for a in attempts)
    assert result.diagnostics.successful_batches == 0
    assert SECRET not in result.diagnostics.model_dump_json()
    assert SECRET not in "\n".join(result.warnings)


def test_recovered_attempt_keeps_first_failure_and_success_without_extra_retry() -> (
    None
):
    llm = FailingLlm(
        LLMStructuredOutputError("Intern response exceeded max_tokens"), recover=True
    )
    result = AutomatedMatrixExtractor(llm, Store()).extract(
        document_id="doc-1", chunks=(_chunk(),)
    )
    attempts = result.diagnostics.batch_attempts
    assert llm.calls == 2
    assert [a.status for a in attempts] == ["error", "ok"]
    assert attempts[0].error_category == "output_truncated"
    assert attempts[1].error_category is None and attempts[1].error_type is None
    assert result.diagnostics.successful_batches == 1
