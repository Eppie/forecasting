from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import mock_open, patch

import pytest

# Provide a minimal 'ollama' module so that src.workflow can be imported
requests_stub = ModuleType("requests")
requests_stub.get = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("requests", requests_stub)

ollama_stub = ModuleType("ollama")


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatResponse:
    message: Message


ollama_stub.Message = Message  # type: ignore[attr-defined]
ollama_stub.ChatResponse = ChatResponse  # type: ignore[attr-defined]
ollama_stub.chat = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("ollama", ollama_stub)

from src import (  # noqa: E402
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
    update_prior,
)
from src.workflow import ReferenceClassItem, get_reference_classes  # noqa: E402


def fake_chat_response(data: object) -> ChatResponse:
    return ChatResponse(message=Message(role="assistant", content=json.dumps(data)))


def test_clarify_question() -> None:
    data = {
        "original_question": "Will AI reach AGI by 2030?",
        "reasoning": "Analyzing trends",
        "clarified_question": "Will AI reach AGI by 2030?",
        "resolution_rule": "official announcement",
        "end_date": "",
        "variable_type": "binary",
    }
    with patch("src.workflow.ollama.chat", return_value=fake_chat_response(data)):
        result = clarify_question("Will AI reach AGI by 2030?")
    assert isinstance(result, Question)
    assert result.original_question == data["original_question"]
    assert result.reasoning == data["reasoning"]
    assert result.resolution_rule == data["resolution_rule"]
    assert result.variable_type == data["variable_type"]
    assert result.clarified_question == data["clarified_question"]


def test_get_reference_classes() -> None:
    q = "Will AI achieve AGI by 2030?"
    data = {"reference_classes": [{"reasoning": "r", "reference_class": "past"}]}
    with patch("src.workflow.ollama.chat", return_value=fake_chat_response(data)) as chat_mock:
        result = get_reference_classes(q)
    chat_mock.assert_called_once()
    assert result == [ReferenceClassItem(reasoning="r", reference_class="past")]


def test_run_workflow_sequence() -> None:
    q = Question(
        original_question="oq",
        reasoning="r",
        resolution_rule="rr",
        end_date="ed",
        variable_type="binary",
        clarified_question="cq",
    )
    base_rates = [BaseRate(reasoning="", reference_class="rc", frequency=0.1)]
    evidence: list[object] = ["e1"]
    prior = 0.2
    probability = 0.3

    with (
        patch("src.workflow.clarify_question", return_value=q) as clarify_mock,
        patch(
            "src.workflow.get_reference_classes",
            return_value=[ReferenceClassItem(reasoning="", reference_class="rc")],
        ) as ref_mock,
        patch("src.workflow.get_base_rates", return_value=base_rates) as base_mock,
        patch("src.workflow.decompose_problem") as decomp_mock,
        patch("src.workflow.gather_evidence", return_value=evidence) as gather_mock,
        patch("src.workflow.update_prior", return_value=prior) as update_mock,
        patch("src.workflow.produce_forecast", return_value=probability) as produce_mock,
        patch("src.workflow.sanity_checks") as sanity_mock,
        patch("src.workflow.cross_validate") as cross_mock,
        patch("src.workflow.record_forecast") as record_mock,
    ):
        result = run_workflow("q")

    clarify_mock.assert_called_once_with("q", False)
    ref_mock.assert_called_once_with(q.clarified_question, False)
    base_mock.assert_called_once_with(q.clarified_question, ref_mock.return_value)
    decomp_mock.assert_called_once_with(q.clarified_question, False)
    gather_mock.assert_called_once_with(q.clarified_question, False)
    update_mock.assert_called_once_with(base_rates, evidence, False)
    produce_mock.assert_called_once_with(prior, False)
    sanity_mock.assert_called_once_with(probability, False)
    cross_mock.assert_called_once_with(probability, False)
    record_mock.assert_called_once_with(q.clarified_question, probability, False)
    assert result == probability


def test_decompose_problem() -> None:
    q = "Will AI achieve AGI by 2030?"
    data = [
        {"driver": "Breakthrough in algorithms", "probability": 0.2},
        {"driver": "Hardware progress", "probability": 0.3},
        {"driver": "Combined", "probability": 0.06},
    ]
    with patch("src.workflow.ollama.chat", return_value=fake_chat_response(data)) as chat_mock:
        result = decompose_problem(q)
    chat_mock.assert_called_once()
    assert result == data


def test_gather_evidence() -> None:
    q = "Will AI achieve AGI by 2030?"
    data = [{"description": "expert comment", "likelihood_ratio": 2.0}]
    with patch("src.workflow.ollama.chat", return_value=fake_chat_response(data)) as chat_mock:
        result = gather_evidence(q)
    chat_mock.assert_called_once()
    assert result == data


def test_update_prior() -> None:
    base_rates = [BaseRate(reasoning="", reference_class="ex", frequency=0.2)]
    evidence = [{"likelihood_ratio": 2.0}, {"likelihood_ratio": 0.5}]
    result = update_prior(base_rates, evidence)
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


def test_record_forecast(tmp_path: Path) -> None:
    q = "t"
    m = mock_open()
    with patch("src.workflow.open", m):
        record_forecast(q, 0.5)
    m.assert_called_once_with("forecasts.jsonl", "a", encoding="utf-8")
    handle = m()
    expected = json.dumps({"question": q, "probability": 0.5}) + "\n"
    handle.write.assert_called_once_with(expected)
