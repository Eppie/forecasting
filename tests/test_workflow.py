from __future__ import annotations

import pytest

from src import (
    BaseRate,
    Question,
    clarify_question,
    cross_validate,
    decompose_problem,
    gather_evidence,
    produce_forecast,
    record_forecast,
    sanity_checks,
    set_base_rate,
    update_prior,
)


def test_workflow_stubs() -> None:
    question_text = "Will AI achieve AGI by 2030?"

    with pytest.raises(NotImplementedError):
        clarify_question(question_text)

    q = Question(text=question_text)

    with pytest.raises(NotImplementedError):
        set_base_rate(q)

    with pytest.raises(NotImplementedError):
        decompose_problem(q)

    with pytest.raises(NotImplementedError):
        gather_evidence(q)

    base_rate = BaseRate(reference_class="example", frequency=0.1)
    with pytest.raises(NotImplementedError):
        update_prior(base_rate, [])

    with pytest.raises(NotImplementedError):
        produce_forecast(0.5)

    with pytest.raises(NotImplementedError):
        sanity_checks(0.5)

    with pytest.raises(NotImplementedError):
        cross_validate(0.5)

    with pytest.raises(NotImplementedError):
        record_forecast(q, 0.5)
