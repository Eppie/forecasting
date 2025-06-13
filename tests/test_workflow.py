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
        "original_question": "Will AI reach AGI by 2030?",
        "reasoning": "Analyzing trends",
        "clarified_question": "Will AI reach AGI by 2030?",
        "resolution_rule": "official announcement",
        "end_date": "",
        "variable_type": "binary",
    }
    mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response(data))

    result = clarify_question("Will AI reach AGI by 2030?")
    assert isinstance(result, Question)
    assert result.original_question == data["original_question"]
    assert result.reasoning == data["reasoning"]
    assert result.resolution_rule == data["resolution_rule"]
    assert result.variable_type == data["variable_type"]
    assert result.clarified_question == data["clarified_question"]


def test_run_workflow_sequence(mocker: MockerFixture) -> None:
    q = Question(
        original_question="oq",
        reasoning="r",
        resolution_rule="rr",
        end_date="ed",
        variable_type="binary",
        clarified_question="cq",
    )
    base_rate = BaseRate(reasoning="", reference_class="rc", frequency=0.1)
    evidence: list[Any] = ["e1"]
    prior = 0.2
    probability = 0.3

    clarify_mock = mocker.patch("src.workflow.clarify_question", return_value=q)
    base_mock = mocker.patch("src.workflow.set_base_rate", return_value=base_rate)
    decomp_mock = mocker.patch("src.workflow.decompose_problem")
    gather_mock = mocker.patch(
        "src.workflow.gather_evidence",
        return_value=evidence,
    )
    update_mock = mocker.patch("src.workflow.update_prior", return_value=prior)
    produce_mock = mocker.patch("src.workflow.produce_forecast", return_value=probability)
    sanity_mock = mocker.patch("src.workflow.sanity_checks")
    cross_mock = mocker.patch("src.workflow.cross_validate")
    record_mock = mocker.patch("src.workflow.record_forecast")

    result = run_workflow("q")

    clarify_mock.assert_called_once_with("q", False)
    base_mock.assert_called_once_with(q.clarified_question, False)
    decomp_mock.assert_called_once_with(q.clarified_question, False)
    gather_mock.assert_called_once_with(q.clarified_question, False)
    update_mock.assert_called_once_with(base_rate, evidence, False)
    produce_mock.assert_called_once_with(prior, False)
    sanity_mock.assert_called_once_with(probability, False)
    cross_mock.assert_called_once_with(probability, False)
    record_mock.assert_called_once_with(q.clarified_question, probability, False)

    assert result == probability


def test_set_base_rate(mocker: MockerFixture) -> None:
    q = "Will AI achieve AGI by 2030?"
    data = {"reference_class": "past AGI timeline predictions"}
    chat_mock = mocker.patch(
        "src.workflow.ollama.chat",
        return_value=fake_chat_response(data),
    )

    result = set_base_rate(q)
    chat_mock.assert_called_once()
    assert isinstance(result, BaseRate)
    assert result.reference_class == data["reference_class"]
    assert result.frequency == 0.0


def test_decompose_problem(mocker: MockerFixture) -> None:
    q = "Will AI achieve AGI by 2030?"
    data = [
        {"driver": "Breakthrough in algorithms", "probability": 0.2},
        {"driver": "Hardware progress", "probability": 0.3},
        {"driver": "Combined", "probability": 0.06},
    ]
    chat_mock = mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response(data))

    result = decompose_problem(q)
    chat_mock.assert_called_once()
    assert result == data


def test_gather_evidence(mocker: MockerFixture) -> None:
    q = "Will AI achieve AGI by 2030?"
    data = [
        {"description": "expert comment", "likelihood_ratio": 2.0},
    ]
    chat_mock = mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response(data))

    result = gather_evidence(q)
    chat_mock.assert_called_once()
    assert result == data


def test_update_prior() -> None:
    base = BaseRate(reasoning="", reference_class="ex", frequency=0.2)
    evidence = [
        {"likelihood_ratio": 2.0},
        {"likelihood_ratio": 0.5},
    ]
    result = update_prior(base, evidence)
    assert result == pytest.approx(0.2)


def test_produce_forecast() -> None:
    expected = 0.12
    assert produce_forecast(0.123) == expected
    with pytest.raises(ValueError):
        produce_forecast(1.2)


def test_sanity_and_cross_validate() -> None:
    sanity_checks(0.5)
    cross_validate(0.5)

    with pytest.raises(ValueError):
        sanity_checks(-0.1)

    with pytest.raises(ValueError):
        cross_validate(1.1)


def test_record_forecast(mocker: MockerFixture) -> None:
    q = "t"
    m = mocker.mock_open()
    open_mock = mocker.patch("src.workflow.open", m)
    record_forecast(q, 0.5)
    open_mock.assert_called_once_with("forecasts.jsonl", "a", encoding="utf-8")
    handle = m()
    expected = json.dumps({"question": q, "probability": 0.5}) + "\n"
    handle.write.assert_called_once_with(expected)
