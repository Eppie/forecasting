from __future__ import annotations

import json
from typing import Any

import ollama
import pytest
from pytest_mock import MockerFixture  # type: ignore

from src import (
    BaseRate,
    Question,
    apply_evidence,
    clarify_question,
    cross_validate,
    decompose_problem,
    gather_evidence,
    produce_forecast,
    record_forecast,
    run_workflow,
    sanity_checks,
    set_base_rate,
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


def test_run_workflow_sequence(mocker: MockerFixture) -> None:
    q = Question(reasoning="r", text="t")
    base_rate = BaseRate(reasoning="br", reference_class="rc", frequency=0.1)
    evidence: list[Any] = ["e1"]
    anchor = 0.2
    probability = {"probability": 0.3, "rationale": "because"}

    clarify_mock = mocker.patch("src.workflow.clarify_question", return_value=q)
    base_mock = mocker.patch("src.workflow.set_base_rate", return_value=base_rate)
    decomp = [{"driver": "combined", "probability": 0.4}]
    decomp_mock = mocker.patch("src.workflow.decompose_problem", return_value=decomp)
    reconcile_mock = mocker.patch("src.workflow.reconcile_views", return_value=anchor)
    gather_mock = mocker.patch("src.workflow.gather_evidence", return_value=evidence)
    apply_mock = mocker.patch("src.workflow.apply_evidence", return_value=0.25)
    produce_mock = mocker.patch("src.workflow.produce_forecast", return_value=probability)
    sanity_mock = mocker.patch("src.workflow.sanity_checks")
    record_mock = mocker.patch("src.workflow.record_forecast")

    result = run_workflow("q")

    clarify_mock.assert_called_once_with("q")
    base_mock.assert_called_once_with(q)
    decomp_mock.assert_called_once_with(q)
    reconcile_mock.assert_called_once_with(base_rate, decomp)
    gather_mock.assert_called_once_with(q)
    apply_mock.assert_called_once_with(anchor, evidence)
    produce_mock.assert_called_once_with(0.25, q)
    sanity_mock.assert_called_once()
    record_mock.assert_called_once_with(q, base_rate, decomp, evidence, probability)

    assert result == probability


def test_set_base_rate(mocker: MockerFixture) -> None:
    q = Question(reasoning="", text="Will AI achieve AGI by 2030?")
    data = {
        "reference_class": "past AGI timeline predictions",
        "frequency": 0.3,
        "reasoning": "analysis",
    }
    chat_mock = mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response(data))

    result = set_base_rate(q)
    chat_mock.assert_called_once()
    assert isinstance(result, BaseRate)
    assert result.reference_class == data["reference_class"]
    assert result.frequency == data["frequency"]
    assert result.reasoning == data["reasoning"]


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


def test_gather_evidence(mocker: MockerFixture) -> None:
    q = Question(reasoning="", text="Will AI achieve AGI by 2030?")
    data = [
        {"description": "expert comment", "likelihood_ratio": 2.0},
    ]
    chat_mock = mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response(data))

    result = gather_evidence(q)
    chat_mock.assert_called_once()
    assert result == data


def test_apply_evidence() -> None:
    start_prob = 0.2
    evidence = [
        {"likelihood_ratio": 2.0},
        {"likelihood_ratio": 0.5},
    ]
    result = apply_evidence(start_prob, evidence)
    assert result == pytest.approx(0.2)


def test_produce_forecast(mocker: MockerFixture) -> None:
    mocker.patch(
        "src.workflow.ollama.chat",
        return_value=fake_chat_response("Rationale"),
    )
    result = produce_forecast(0.123, Question(reasoning="", text="q"))
    expected_prob = 0.12
    assert result["probability"] == expected_prob


def test_sanity_and_cross_validate(mocker: MockerFixture) -> None:
    base = BaseRate(reasoning="r", reference_class="rc", frequency=0.5)
    question = Question(reasoning="r", text="t")
    decomposition = [
        {"driver": "combined", "probability": 0.5},
    ]
    evidence: list[Any] = []
    mocker.patch("src.workflow.ollama.chat", return_value=fake_chat_response("critique"))
    sanity_checks(0.5, base, question, decomposition, evidence)
    cross_validate(0.5)

    with pytest.raises(ValueError):
        cross_validate(1.1)


def test_record_forecast(mocker: MockerFixture) -> None:
    q = Question(reasoning="r", text="t")
    base = BaseRate(reasoning="br", reference_class="rc", frequency=0.1)
    decomposition: list[Any] = []
    evidence: list[Any] = []
    final = {"probability": 0.5, "rationale": "r"}
    m = mocker.mock_open()
    open_mock = mocker.patch("src.workflow.open", m)
    record_forecast(q, base, decomposition, evidence, final)
    open_mock.assert_called_once_with("forecasts.jsonl", "a", encoding="utf-8")
    handle = m()
    written = json.loads(handle.write.call_args.args[0])
    assert written["final_forecast"] == final
