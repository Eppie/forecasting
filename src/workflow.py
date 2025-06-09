"""Stub definitions for the forecasting workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import ollama


@dataclass
class Question:
    """Forecasting question details."""

    reasoning: str
    text: str
    resolution_rule: str | None = None
    variable_type: str | None = None


def clarify_question(question: str) -> Question:
    """Return a ``Question`` object from raw text.

    Args:
        question: The question to clarify.

    Returns:
        A ``Question`` instance.
    """
    system_prompt = (
        "Clarify the following forecasting question."
        " Provide JSON with fields 'question', 'reasoning',"
        " 'resolution_rule', and 'variable_type'."
    )

    response = ollama.chat(
        model="llama3.3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        format="json",
        options={"temperature": 0},
    )

    content = response.message.content
    if content is None:
        raise ValueError("Model returned empty content")
    data = json.loads(content)

    reasoning = str(data.get("reasoning", ""))
    text = str(data.get("question", question))
    resolution_rule = data.get("resolution_rule")
    variable_type = data.get("variable_type")

    return Question(
        reasoning=reasoning,
        text=text,
        resolution_rule=resolution_rule,
        variable_type=variable_type,
    )


@dataclass
class BaseRate:
    """Base rate prior information."""

    reference_class: str
    frequency: float


def set_base_rate(question: Question) -> BaseRate:
    """Determine the base rate for a question.

    Step 1 identifies a suitable reference class for the question using an LLM.
    Step 2 (measuring the historical frequency) is not yet implemented and the
    returned ``BaseRate`` therefore uses ``0.0`` as a placeholder for
    ``frequency``.

    Args:
        question: The clarified forecasting question.

    Returns:
        A :class:`BaseRate` with the reference class filled in and ``frequency``
        set to ``0.0`` until the next step is implemented.
    """

    system_prompt = (
        "Suggest an appropriate reference class for the following forecasting "
        "question. Return JSON with a single field 'reference_class'."
    )

    response = ollama.chat(
        model="llama3.3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.text},
        ],
        format="json",
        options={"temperature": 0},
    )

    content = response.message.content
    if content is None:
        raise ValueError("Model returned empty content")

    data = json.loads(content)
    reference_class = str(data.get("reference_class", ""))

    # TODO: implement measurement of historical frequency based on the chosen
    # reference class.
    frequency = 0.0

    return BaseRate(reference_class=reference_class, frequency=frequency)


def decompose_problem(question: Question) -> list[Any]:
    """Break the question into smaller drivers."""
    raise NotImplementedError


def gather_evidence(question: Question) -> list[Any]:
    """Collect evidence relevant to the question."""
    raise NotImplementedError


def update_prior(base_rate: BaseRate, evidence: list[Any]) -> float:
    """Update the prior probability based on evidence."""
    raise NotImplementedError


def produce_forecast(probability: float) -> float:
    """Produce the final forecast probability."""
    raise NotImplementedError


def sanity_checks(probability: float) -> None:
    """Perform sanity and bias checks on the forecast."""
    raise NotImplementedError


def cross_validate(probability: float) -> None:
    """Optional cross-validation with external sources."""
    raise NotImplementedError


def record_forecast(question: Question, probability: float) -> None:
    """Record the forecast and related metadata."""
    raise NotImplementedError


def run_workflow(question_text: str) -> Question:
    """Run the forecasting workflow for a single question."""

    return clarify_question(question_text)
