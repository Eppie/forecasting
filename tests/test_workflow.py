from __future__ import annotations

import json
from typing import Any

import ollama
import pytest
from pytest_mock import MockerFixture  # type: ignore

from src import (
    BaseRate,
    Question,
    clarify_question,
    cross_validate,
    decompose_problem,
    gather_evidence,
    produce_forecast,
    record_forecast,
    run_workflow,
    sanity_checks,
    set_base_rate,
    update_prior,
)


def fake_chat_response(data: dict[str, Any]) -> ollama.ChatResponse:
    return ollama.ChatResponse(message=ollama.Message(role="assistant", content=json.dumps(data)))


def test_clarify_question(mocker: MockerFixture) -> None:
    data = {
        "question": "Will AI reach AGI by 2030?",
        "reasoning": "Analyzing trends",
        "resolution_rule": "official announcement",
        "variable_type": "binary",
    }
    mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response(data))

    result = clarify_question("Will AI reach AGI by 2030?")
    assert isinstance(result, Question)
    assert result.text == data["question"]
    assert result.reasoning == data["reasoning"]
    assert result.resolution_rule == data["resolution_rule"]
    assert result.variable_type == data["variable_type"]


def test_run_workflow_calls_clarify(mocker: MockerFixture) -> None:
    q = Question(reasoning="r", text="t")
    clarify_mock = mocker.patch("src.workflow.clarify_question", return_value=q)

    result = run_workflow("q")
    clarify_mock.assert_called_once_with("q")
    assert result is q


def test_other_stubs() -> None:
    q = Question(reasoning="", text="Will AI achieve AGI by 2030?")

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
