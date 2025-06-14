from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import ollama

from base_rates import get_base_rates  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Forecasting question details."""

    original_question: str
    reasoning: str
    resolution_rule: str
    end_date: str
    variable_type: str
    clarified_question: str


def clarify_question(question: str, verbose: bool = False) -> Question:
    """Return a ``Question`` object from raw text.

    1.1 Write the question verbatim.
        Record the wording, date, and who asked it so you can track resolution later.
    1.2 Define an unambiguous resolution rule.
        Specify what counts as "yes," "no," or the numeric outcome, the exact end-date,
        and a publicly checkable source (e.g., "Trump assassinated" = death caused by violence,
        confirmed by two major newswires, any time before noon EST 20 Jan 2029).
    1.3 Classify the variable type. Binary → probability; continuous → predictive distribution.

    Args:
        question: The question to clarify.
        verbose: Whether to enable verbose output.

    Returns:
        A ``Question`` instance.
    """
    system_prompt = """
        You are a senior super-forecaster.  Your task is to execute Steps 1.1–1.3
        of the Good-Judgment workflow on a raw forecasting prompt and output
        **ONLY valid JSON**.

        Return an object with these keys (all required, no extras):

        • original_question  – verbatim copy of the user's text
        • reasoning          – detailed chain‑of‑thought (≈3–6 sentences) explaining how you derived the clarification and why each field is appropriate.
        • resolution_rule    – bullet-style rule that states what counts as YES/NO
          (or how the numeric outcome will be judged), the exact end date-time,
          and at least one publicly checkable source
        • end_date           – ISO-8601 date-time, e.g. "2029-01-20T17:00:00Z"
        • variable_type      – one of "binary", "count", "continuous"
        • clarified_question – rewritten so the outcome, metric, population, and
          closing date are explicit and testable

        ### Few-shot examples

        1. Trump assassination
        {
          "original_question": "Will Donald Trump be assassinated before the end of his presidency?",
          "reasoning": "Defined 'assassinated', fixed the constitutional end-time.",
          "resolution_rule": "YES if two major newswires confirm Trump's death caused by a hostile act before the deadline; NO otherwise.",
          "end_date": "2029-01-20T17:00:00Z",
          "variable_type": "binary"
          "clarified_question": "Will Donald Trump die due to a hostile human act before 2029-01-20 17:00 UTC?",
        }

        2. S&P 500 year‑end level
        {
          "original_question": "Where will the S&P 500 close on 31 Dec 2025?",
          "reasoning": "Identified the forecast variable as the official S&P 500 index closing value on the final trading day of 2027; clarified the observation date and the data source (S&P Dow Jones Indices). Explained that a continuous numeric outcome requires a distribution rather than a single probability.",
          "resolution_rule": "Use the closing value published by S&P Dow Jones Indices for the S&P 500 on 2025‑12‑31 (or the next trading day if markets are closed) as reported by Bloomberg or the Wall Street Journal.",
          "end_date": "2026-01-02T16:00:00Z",
          "variable_type": "continuous",
          "clarified_question": "What will the official closing value of the S&P 500 index be on the final trading day of 2025?"
        }

        3. Cat-5 hurricanes
        {
          "original_question": "How many Category 5 Atlantic hurricanes will there be in 2026?",
          "reasoning": "Specified basin, intensity threshold, and season window.",
          "resolution_rule": "Count storms designated Category 5 by NOAA’s Best Track dataset as of 2027-03-01.",
          "end_date": "2027-03-01T00:00:00Z",
          "variable_type": "count"
          "clarified_question": "How many Atlantic basin tropical cyclones will reach Category 5 (≥157 mph) between 2026-06-01 and 2026-11-30?",
        }

        ### Instructions
        1. Think step‑by‑step but return only the final JSON. Provide a *thorough* reasoning field as specified.
        2. Choose exactly one variable_type.
        3. Use UTC for the end_date.
        """

    response = ollama.chat(
        model="deepseek-r1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        format="json",
    )

    content = response.message.content
    if content is None:
        raise ValueError("Model returned empty content")
    data = json.loads(content)

    original_question = data.get("original_question", question)
    reasoning = data.get("reasoning", "")
    resolution_rule = data.get("resolution_rule")
    end_date = data.get("end_date", "")
    variable_type = data.get("variable_type")
    clarified_question = data.get("clarified_question", "")

    result = Question(
        original_question=original_question,
        reasoning=reasoning,
        resolution_rule=resolution_rule,
        end_date=end_date,
        variable_type=variable_type,
        clarified_question=clarified_question,
    )
    logger.debug("%s", result)
    return result


@dataclass
class BaseRate:
    """Base rate prior information."""

    reasoning: str
    reference_class: str
    frequency: float


@dataclass
class ReferenceClassItem:
    """Single candidate reference class with justification."""

    reasoning: str
    reference_class: str


def get_reference_classes(clarified_question: str, verbose: bool = False) -> list[ReferenceClassItem]:
    """
    Step 2.1 — Identify reference classes.

    A reference class is a historical set of events that share causal structure
    with the forecast target. This function asks an LLM to suggest 2–4 candidate
    reference classes, each with a justification.

    Args:
        clarified_question: The clarified forecasting question (output of Step 1).
        verbose: If True, pretty‑print the result.

    Returns:
        A list of ReferenceClassItem (each with reasoning and reference_class).
    """

    system_prompt = """
    You are an expert super‑forecaster performing **Step 2.1 – Identify reference classes**.
    Given the clarified forecasting question below, output **ONLY valid JSON**: a top-level key called "reference_classes",
        which has a value that is a JSON array of 2–4 objects.
    Each object must have:
    • reasoning        – 2–4 sentences explaining why your chosen class shares
                         causal structure with the target event.
    • reference_class  – a concise label for the set (e.g.,
                         "U.S. presidents assassinated vs total presidents").

    ### Few‑shot examples

    1. Trump assassination
    { "reference_classes": [
      {
        "reasoning": "The risk for any sitting president is best benchmarked against all completed U.S. presidencies because they share identical constitutional duties, security context, and public exposure.",
        "reference_class": "U.S. presidents assassinated vs total presidents"
      },
      {
        "reasoning": "Assassination attempts on major world leaders with similar security protocols provide further context.",
        "reference_class": "Assassination attempts on G7 heads of government, 1945–present"
      }
    ] }

    2. S&P 500 year‑end level
    { "reference_classes": [
      {
        "reasoning": "Historical year‑end closing values of the S&P 500 capture long‑run market behaviour under varied macro conditions and thus form an appropriate statistical backdrop for the forecast.",
        "reference_class": "Distribution of S&P 500 year‑end closes, 1974‑2023"
      },
      {
        "reasoning": "Year‑end closes of major global equity indices provide a broader perspective on market cycles.",
        "reference_class": "Year‑end closes for MSCI World Index, 1980–2023"
      }
    ] }

    3. Cat‑5 hurricanes
    { "reference_classes": [
      {
        "reasoning": "NOAA Best‑Track records of Category 5 Atlantic storms provide direct historical instances of the phenomenon of interest.",
        "reference_class": "Atlantic Category 5 hurricanes 1924‑2024"
      },
      {
        "reasoning": "Atlantic hurricanes of Category 4 and above, as they share similar meteorological drivers and impacts.",
        "reference_class": "Atlantic Category 4–5 hurricanes 1924‑2024"
      }
    ] }

    ### Instructions
    1. Think step‑by‑step but return **only** the final JSON array.
    2. The array MUST contain 2–4 objects, each with both fields (no empty fields).
    3. It's okay if you are unsure, just reason your way through to an answer.
    """

    response = ollama.chat(
        model="deepseek-r1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clarified_question},
        ],
        format="json",
    )

    content = response.message.content
    if content is None:
        raise ValueError("Model returned empty content")
    data = json.loads(content)
    logger.debug("%s", data)
    items = [ReferenceClassItem(**item) for item in data["reference_classes"]]
    logger.debug("%s", items)
    return items


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


def decompose_problem(clarified_question: str, verbose: bool = False) -> list[ProblemDecomposition]:
    """Break the question into smaller drivers.

    This step constructs an "inside view" by asking the model to:
        3.1	Break the event into independent drivers
            ("Could a shooter get close?" × "Security failure?" × "Medical non-survival?").  ￼
        3.2	Assign rough probabilities or ranges to each piece using back-of-the-envelope logic.  ￼
        3.3	Recombine (usually by multiplication or scenario trees)
            to create an inside-view estimate that you will compare against the base rate.  ￼

    The model is expected to return JSON where each element describes a
    driver and its probability. The final element should represent the
    combined inside-view probability.

    Args:
        clarified_question: The clarified forecasting question.
        verbose: Whether to enable verbose output.

    Returns:
        A list of driver descriptions with probabilities. The last element
        represents the combined inside-view estimate.
    """

    system_prompt = """
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

    Break the event into independent drivers ("Could a shooter get close?'
     × "Security failure?" × "Medical non-survival?"). '
    Assign rough probabilities or ranges to each piece using back-of-the-envelope
     logic. Recombine (usually by multiplication or scenario trees) to create
     an inside-view estimate that you will compare against the base rate.
     Return the result as a JSON object of type ProblemDecomposition.
     The user will now provide the question for you to decompose into drivers.
    """

    response = ollama.chat(
        model="deepseek-r1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clarified_question},
        ],
        format="json",
    )

    content = response.message.content
    if content is None:
        raise ValueError("Model returned empty content")

    data = json.loads(content)

    logger.debug("%s", data)
    if not isinstance(data, list):
        raise ValueError("Model did not return a list")

    return data


def gather_evidence(clarified_question: str, verbose: bool = False) -> list[Any]:
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
        clarified_question: The clarified forecasting question.
        verbose: Whether to enable verbose output.

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
        model="deepseek-r1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clarified_question},
        ],
        format="json",
    )

    content = response.message.content
    if content is None:
        raise ValueError("Model returned empty content")

    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("Model did not return a list")

    return data


def update_prior(base_rates: list[BaseRate], evidence: list[Any], verbose: bool = False) -> float:
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
        base_rates: The base rates prior information.
        evidence: A list of evidence dictionaries.
        verbose: Whether to enable verbose output.

    Returns:
        The posterior probability after applying all likelihood ratios.
    """

    probabilities = []
    for base_rate in base_rates:
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
        probabilities.append(probability)

    return sum(probabilities) / len(probabilities)


def produce_forecast(probability: float, verbose: bool = False) -> float:
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
        verbose: Whether to enable verbose output.

    Returns:
        The rounded probability.
    """

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    return round(probability, 2)


def sanity_checks(probability: float, verbose: bool = False) -> None:
    """Perform sanity and bias checks on the forecast.

    7.1 Check against the base-rate anchor. If you moved > 4× in odds, be ready to justify.
        7.2 Overconfidence sweep. Ask: Would I bet money at these odds?
        7.3 Common cognitive traps checklist: availability, confirmation, wishful thinking.

    This simplified version only verifies that the probability is within ``[0, 1]``.
    In a production system additional checks for common cognitive biases would be performed.

    Args:
        probability: Probability to validate.
        verbose: Whether to enable verbose output.

    Raises:
        ValueError: If ``probability`` lies outside the allowed range.
    """

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    # No further action in this stub implementation.


def cross_validate(probability: float, verbose: bool = False) -> None:
    """Optional cross-validation with external sources.
    8.1 Compare with prediction-market prices or crowd forecasts.
    8.2 Score hypothetical accuracy (Brier) vs. alternative estimates for robustness.

    Args:
        probability: Probability to validate.
        verbose: Whether to enable verbose output.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    # Placeholder for external cross-validation logic.


def record_forecast(clarified_question: str, probability: float, verbose: bool = False) -> None:
    """Record the forecast and related metadata.

    9.1 The final forecast and date.
    9.2 Key assumptions, data sources, and Fermi breakdown.

    The forecast is appended as a JSON line to ``forecasts.jsonl`` in the current
    working directory. Each line contains the question text and the probability
    value.

    Args:
        clarified_question: The clarified question being answered.
        probability: The final forecast probability.
        verbose: Whether to enable verbose output.
    """

    entry = {"question": clarified_question, "probability": probability}
    with open("forecasts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_workflow(question_text: str, verbose: bool = False) -> float:
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
        verbose: Whether to enable verbose output.

    Returns:
        The final forecast probability.
    """

    question = clarify_question(question_text, verbose)
    clarified_question = question.clarified_question
    # clarified_question = "What will be OpenAI's total revenue for the fiscal year ending December 31, 2027?"
    ref_classes = get_reference_classes(clarified_question, verbose)
    # TODO: pass one of these to get_base_rate when implemented
    base_rates = get_base_rates(clarified_question, ref_classes)

    decompose_problem(clarified_question, verbose)
    evidence = gather_evidence(clarified_question, verbose)
    # The following will error unless base_rate is set, but we leave the workflow structure.
    prior = update_prior(base_rates, evidence, verbose)
    probability = produce_forecast(prior, verbose)
    sanity_checks(probability, verbose)
    cross_validate(probability, verbose)
    record_forecast(question.clarified_question, probability, verbose)
    return probability
