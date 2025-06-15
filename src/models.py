from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Question:
    """Forecasting question details."""

    original_question: str
    reasoning: str
    resolution_rule: str
    end_date: str
    variable_type: str
    clarified_question: str


@dataclass
class ReferenceClassItem:
    """Single candidate reference class with justification."""

    reasoning: str
    reference_class: str


@dataclass
class BaseRate:
    """Base rate prior information."""

    reference_class: str
    reasoning: str
    numerator: int | None = None
    denominator: int | None = None
    frequency: float | None = None  # 0–1 for binary / proportion
    distribution: dict[str, float] | None = None  # median/p5/p95 for continuous
    lambda_: float | None = None  # Poisson rate for discrete counts
    quality_score: float | None = None  # crude 0–1 confidence


@dataclass
class ContinuousDriver:
    driver: str
    low_value: float
    high_value: float


@dataclass
class DiscreteDriver:
    driver: str
    probability: float


@dataclass
class ProblemDecomposition:
    reasoning: str
    drivers: list[ContinuousDriver | DiscreteDriver]
