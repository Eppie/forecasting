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

    1.1 Write the question verbatim.
        Record the wording, date, and who asked it so you can track resolution later.
    1.2 Define an unambiguous resolution rule.
        Specify what counts as “yes,” “no,” or the numeric outcome, the exact end-date,
        and a publicly checkable source (e.g., “Trump assassinated” = death caused by violence,
        confirmed by two major newswires, any time before noon EST 20 Jan 2029).
    1.3 Classify the variable type. Binary → probability; continuous → predictive distribution.

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

    2.1 Identify a reference class.
        Choose past events that are similar enough to share causal structure
        (e.g., sitting U.S. presidents killed in office).  ￼
        2.2	Measure historical frequency or distribution.
            For binaries, convert to a %; for numerics, fit a simple distribution (median, 5th/95th-percentiles).  ￼
        2.3	Write down that number—this is your prior.
            Resist the urge to tweak it yet; superforecasters explicitly anchor on the base rate first.

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
    """Break the question into smaller drivers.

    This step constructs an "inside view" by asking the model to:
        3.1	Break the event into independent drivers
            (“Could a shooter get close?” × “Security failure?” × “Medical non-survival?”).  ￼
        3.2	Assign rough probabilities or ranges to each piece using back-of-the-envelope logic.  ￼
        3.3	Recombine (usually by multiplication or scenario trees)
            to create an inside-view estimate that you will compare against the base rate.  ￼

    The model is expected to return JSON where each element describes a
    driver and its probability. The final element should represent the
    combined inside-view probability.

    Args:
        question: The clarified forecasting question.

    Returns:
        A list of driver descriptions with probabilities. The last element
        represents the combined inside-view estimate.
    """

    system_prompt = (
        'Break the event into independent drivers ("Could a shooter get close?"'
        ' × "Security failure?" × "Medical non-survival?"). '
        "Assign rough probabilities or ranges to each piece using back-of-the-envelope"
        " logic. Recombine (usually by multiplication or scenario trees) to create"
        " an inside-view estimate that you will compare against the base rate."
        " Return the result as a JSON list of objects with fields 'driver' and"
        " 'probability'. Include a final object with driver 'combined' containing"
        " the inside-view probability."
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
    if not isinstance(data, list):
        raise ValueError("Model did not return a list")

    return data


def gather_evidence(question: Question) -> list[Any]:
    """Collect evidence relevant to the question.

    4.1 Quick desk research.
        Newsfeeds, SEC filings, event databases. Capture facts that materially change odds.
    4.2 High-value data points.
        Polls, market prices, expert testimony. Note likelihood ratios (how much more/less likely if fact is true?).
    4.3 Reliability check.
        Source reputation, recency. Tag each item with a credibility weight.

    This implementation relies on an LLM to return a concise list of evidence
    items. Each evidence item should include a short description and may
    optionally contain a ``likelihood_ratio`` field which is used later when
    updating the prior.

    Args:
        question: The clarified forecasting question.

    Returns:
        A list of evidence dictionaries.
    """

    system_prompt = (
        "List key pieces of evidence that would influence the probability of the"
        " following forecasting question. Return the result as a JSON list of"
        " objects. Each object should include a 'description' field and may"
        " include a numeric 'likelihood_ratio'."
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
    if not isinstance(data, list):
        raise ValueError("Model did not return a list")

    return data


def update_prior(base_rate: BaseRate, evidence: list[Any]) -> float:
    """Update the prior probability based on evidence.

    5.1 Translate each piece of evidence into a likelihood ratio (formal Bayes or an intuitive ×/÷).
        5.2 Apply sequential updating to move your prior toward the inside-view figure from Step 3.
        5.3 Document the math in two lines:
            Prior → Posterior for binaries, or Prior distribution → Posterior
            distribution/credible interval for numerics.

    Evidence items may contain a ``likelihood_ratio`` key representing how much
    the evidence shifts the odds. The prior is updated sequentially using these
    ratios. Items lacking the key are ignored.

    Args:
        base_rate: The base rate prior information.
        evidence: A list of evidence dictionaries.

    Returns:
        The posterior probability after applying all likelihood ratios.
    """

    probability = base_rate.frequency
    for item in evidence:
        if isinstance(item, dict) and "likelihood_ratio" in item:
            lr = float(item["likelihood_ratio"])
            # Convert probability to odds, apply likelihood ratio, then
            # convert back to probability.
            if probability in (0.0, 1.0):
                odds = float("inf") if probability == 1.0 else 0.0
            else:
                odds = probability / (1 - probability)
            odds *= lr
            probability = odds / (1 + odds) if odds != float("inf") else 1.0

    return probability


def produce_forecast(probability: float) -> float:
    """Produce the final forecast probability.

    6.1 Binary Question
        Report: single probability rounded to the nearest whole % (e.g., 11 %) and one-sentence rationale.

    6.2 Continuous Question
        Report: central range (e.g., 10th–90th or 5th–95th) plus a best-guess median/mean
        (e.g., $20 B–$100 B, 90 % CI; median ≈ $50 B). Indicate any skew.
        (Superforecasters favour log-normal or student-t for heavy-tailed economic variables.)

    For this simplified implementation only binary questions are supported. The
    probability is validated to be between ``0`` and ``1`` and then rounded to
    two decimal places.

    Args:
        probability: Posterior probability from :func:`update_prior`.

    Returns:
        The rounded probability.
    """

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    return round(probability, 2)


def sanity_checks(probability: float) -> None:
    """Perform sanity and bias checks on the forecast.

    7.1 Check against the base-rate anchor. If you moved > 4× in odds, be ready to justify.
        7.2 Overconfidence sweep. Ask: Would I bet money at these odds?
        7.3 Common cognitive traps checklist: availability, confirmation, wishful thinking.

    This simplified version only verifies that the probability is within ``[0, 1]``.
    In a production system additional checks for common cognitive biases would be performed.

    Args:
        probability: Probability to validate.

    Raises:
        ValueError: If ``probability`` lies outside the allowed range.
    """

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    # No further action in this stub implementation.


def cross_validate(probability: float) -> None:
    """Optional cross-validation with external sources.
    8.1 Compare with prediction-market prices or crowd forecasts.
        8.2 Score hypothetical accuracy (Brier) vs. alternative estimates for robustness.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    # Placeholder for external cross-validation logic.


def record_forecast(question: Question, probability: float) -> None:
    """Record the forecast and related metadata.

    9.1 The final forecast and date.
        9.2 Key assumptions, data sources, and Fermi breakdown.

    The forecast is appended as a JSON line to ``forecasts.jsonl`` in the current
    working directory. Each line contains the question text and the probability
    value.

    Args:
        question: The clarified question being answered.
        probability: The final forecast probability.
    """

    entry = {"question": question.text, "probability": probability}
    with open("forecasts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_workflow(question_text: str) -> float:
    """Run the forecasting workflow for a single question.

    The function executes each step of the forecasting workflow in order:

    1. Clarify the question.
    2. Determine the base rate.
    3. Decompose the problem into drivers.
    4. Gather current evidence.
    5. Update the prior based on the evidence.
    6. Produce the forecast.
    7. Run sanity and bias checks.
    8. Optionally cross‑validate.
    9. Record the forecast.

    Args:
        question_text: The raw forecasting question.

    Returns:
        The final forecast probability.
    """

    question = clarify_question(question_text)
    base_rate = set_base_rate(question)
    decompose_problem(question)
    evidence = gather_evidence(question)
    prior = update_prior(base_rate, evidence)
    probability = produce_forecast(prior)
    sanity_checks(probability)
    cross_validate(probability)
    record_forecast(question, probability)
    return probability
