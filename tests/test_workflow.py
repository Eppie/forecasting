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


def fake_chat_response(data: Any) -> ollama.ChatResponse:
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


def test_set_base_rate(mocker: MockerFixture) -> None:
    q = Question(reasoning="", text="Will AI achieve AGI by 2030?")
    data = {"reference_class": "past AGI timeline predictions"}
    chat_mock = mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response(data))

    result = set_base_rate(q)
    chat_mock.assert_called_once()
    assert isinstance(result, BaseRate)
    assert result.reference_class == data["reference_class"]
    assert result.frequency == 0.0


def test_decompose_problem(mocker: MockerFixture) -> None:
    q = Question(reasoning="", text="Will AI achieve AGI by 2030?")
    data = [
        {"driver": "Breakthrough in algorithms", "probability": 0.2},
        {"driver": "Hardware progress", "probability": 0.3},
        {"driver": "Combined", "probability": 0.06},
    ]
    chat_mock = mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response(data))

    result = decompose_problem(q)
    chat_mock.assert_called_once()
    assert result == data


def test_other_stubs() -> None:
    q = Question(reasoning="", text="Will AI achieve AGI by 2030?")

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
