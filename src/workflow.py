"""Stub definitions for the forecasting workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Question:
    """Forecasting question details."""

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
    raise NotImplementedError


@dataclass
class BaseRate:
    """Base rate prior information."""

    reference_class: str
    frequency: float


def set_base_rate(question: Question) -> BaseRate:
    """Determine the base rate for a question."""
    raise NotImplementedError


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
