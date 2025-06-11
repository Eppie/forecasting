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

    reasoning: str
    reference_class: str
    frequency: float


def set_base_rate(question: Question) -> BaseRate:
    """Determine the base rate for a question by finding a reference class and its historical frequency."""

    system_prompt = (
        "You are an expert forecaster. For the given question, determine the 'outside view' base rate. "
        "First, identify a suitable reference class of similar past events. "
        "Second, based on historical data, state the frequency of the outcome in that class. "
        "Provide your reasoning for this choice. "
        "Return JSON with fields 'reference_class', 'frequency', and 'reasoning'."
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
    frequency = float(data.get("frequency", 0.5))
    reasoning = str(data.get("reasoning", ""))

    if not reference_class or not reasoning:
        raise ValueError("Model failed to return a valid reference class and reasoning.")

    return BaseRate(reasoning=reasoning, reference_class=reference_class, frequency=frequency)


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


def reconcile_views(base_rate: BaseRate, inside_view_decomposition: list[Any]) -> float:
    """Reconcile the outside and inside views to create an anchor probability."""

    inside_view_prob = 0.5
    for item in inside_view_decomposition:
        if isinstance(item, dict) and item.get("driver") == "combined":
            inside_view_prob = float(item.get("probability", 0.5))
            break
    else:
        print("Warning: 'combined' driver not found in decomposition. Defaulting inside view to 0.5.")

    outside_view_prob = base_rate.frequency
    reconciled_prob = (outside_view_prob + inside_view_prob) / 2

    print(
        "Reconciling Views: Outside View (Base Rate): "
        f"{outside_view_prob:.2%}, Inside View (Decomposition): {inside_view_prob:.2%}"
    )
    print(f"Reconciled Anchor Probability: {reconciled_prob:.2%}")

    return reconciled_prob


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


def apply_evidence(start_probability: float, evidence: list[Any]) -> float:
    """Update a starting probability based on evidence with likelihood ratios.

    5.1 Translate each piece of evidence into a likelihood ratio (formal Bayes or an intuitive ×/÷).
        5.2 Apply sequential updating to move your prior toward the inside-view figure from Step 3.
        5.3 Document the math in two lines:
            Prior → Posterior for binaries, or Prior distribution → Posterior
            distribution/credible interval for numerics.

    Evidence items may contain a ``likelihood_ratio`` key representing how much
    the evidence shifts the odds. The prior is updated sequentially using these
    ratios. Items lacking the key are ignored.

    Args:
        start_probability: The anchor probability.
        evidence: A list of evidence dictionaries.

    Returns:
        The posterior probability after applying all likelihood ratios.
    """

    probability = start_probability
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


def produce_forecast(probability: float, question: Question) -> dict[str, Any]:
    """Produce the final forecast with a one-sentence rationale.

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
        probability: The final posterior probability.
        question: The clarified question.

    Returns:
        A dictionary with the final probability and a rationale.
    """

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    system_prompt = (
        "You are a forecasting analyst. Given the final probability for the question, "
        "write a concise, one-sentence rationale explaining the forecast. "
        "Report the probability rounded to the nearest whole percent."
    )
    context = f"Question: {question.text}\nFinal Probability: {probability:.3f}"

    response = ollama.chat(
        model="llama3.3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ],
    )

    rationale = response.message.content or "Rationale could not be generated."

    final_prob_rounded = round(probability, 2)

    return {"probability": final_prob_rounded, "rationale": rationale.strip()}


def sanity_checks(
    probability: float,
    base_rate: BaseRate,
    question: Question,
    decomposition: list[Any],
    evidence: list[Any],
) -> None:
    """Perform sanity and bias checks using an LLM as a "red team"."""

    base_rate_odds = base_rate.frequency / (1 - base_rate.frequency)
    final_odds = probability / (1 - probability)
    if final_odds > base_rate_odds * 4 or final_odds < base_rate_odds / 4:
        print(
            "Warning: Final odds "
            f"({final_odds:.2f}) have shifted by more than 4x "
            f"from the base rate odds ({base_rate_odds:.2f})."
        )

    system_prompt = (
        "You are a devil's advocate and expert in cognitive biases. "
        "Review the following forecast and challenge it. "
        "Specifically, check for: \n"
        "1.  **Availability Heuristic:** Is the forecast overly influenced by recent, vivid news? \n"
        "2.  **Confirmation Bias:** Does the evidence search look one-sided? \n"
        "3.  **Wishful Thinking:** Could the forecaster's desires be influencing the probability? \n"
        "4.  **Overconfidence:** Is the forecast too extreme? Ask 'Would I bet my own money at these odds?' \n"
        "Provide a short, critical review pointing out the single biggest potential flaw or bias."
    )

    context = (
        f"Forecasting Question: {question.text}\n"
        f"Initial Base Rate (Outside View): {base_rate.frequency:.2%} ({base_rate.reference_class})\n"
        f"Inside View Decomposition: {json.dumps(decomposition, indent=2)}\n"
        f"Key Evidence Considered: {json.dumps(evidence, indent=2)}\n"
        f"---"
        f"Final Forecast Probability: {probability:.2%}\n"
        f"---"
        f"Critique this forecast based on the instructions."
    )

    response = ollama.chat(
        model="llama3.3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ],
        options={"temperature": 0.5},
    )

    critique = response.message.content
    print("\n--- Sanity & Bias Check ---")
    if critique:
        print(critique)
    print("---------------------------\n")


def cross_validate(probability: float) -> None:
    """Optional cross-validation with external sources.
    8.1 Compare with prediction-market prices or crowd forecasts.
        8.2 Score hypothetical accuracy (Brier) vs. alternative estimates for robustness.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    # Placeholder for external cross-validation logic.


def record_forecast(
    question: Question,
    base_rate: BaseRate,
    decomposition: list[Any],
    evidence: list[Any],
    final_forecast: dict[str, Any],
) -> None:
    """Record the full forecast and its components to a file."""

    import datetime

    entry = {
        "question": question.text,
        "resolution_rule": question.resolution_rule,
        "forecast_date": datetime.datetime.now().isoformat(),
        "final_forecast": final_forecast,
        "components": {
            "base_rate": {
                "reasoning": base_rate.reasoning,
                "reference_class": base_rate.reference_class,
                "frequency": base_rate.frequency,
            },
            "decomposition": decomposition,
            "evidence": evidence,
        },
    }
    with open("forecasts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_workflow(question_text: str) -> dict[str, Any]:
    """Run the full superforecasting workflow for a single question.

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
    print(f"1. Clarified Question: '{question.text}' ({question.variable_type})")

    base_rate = set_base_rate(question)
    print(f"2. Base Rate (Outside View): {base_rate.frequency:.2%} from class '{base_rate.reference_class}'")

    decomposition = decompose_problem(question)
    print("3. Problem Decomposed (Inside View)")

    reconciled_anchor = reconcile_views(base_rate, decomposition)

    evidence = gather_evidence(question)
    print(f"5. Gathered {len(evidence)} pieces of new evidence.")

    posterior_prob = apply_evidence(reconciled_anchor, evidence)
    print(f"6. Probability updated to {posterior_prob:.2%} after evidence.")

    sanity_checks(posterior_prob, base_rate, question, decomposition, evidence)

    final_forecast = produce_forecast(posterior_prob, question)
    print(f"8. Final Forecast: {final_forecast['probability']:.0%} - {final_forecast['rationale']}")

    record_forecast(question, base_rate, decomposition, evidence, final_forecast)
    print("9. Forecast recorded to forecasts.jsonl")

    return final_forecast
