"""
TED-LW Dataset Module
=====================
Loads GAIR/LIMO, extracts questions/answers, and grades model outputs.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

from datasets import load_dataset

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_limo(max_problems: Optional[int] = None) -> List[Dict]:
    """
    Load the GAIR/LIMO dataset and return a list of problem dicts.

    Each dict has keys:
        - "problem_id": int (0-indexed)
        - "question": str
        - "answer": str (ground-truth answer, cleaned)
        - "raw": dict (original dataset row for reference)
    """
    logger.info(f"Loading dataset {config.DATASET_NAME} (split={config.DATASET_SPLIT})")
    ds = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    logger.info(f"Dataset loaded: {len(ds)} samples")
    logger.info(f"Dataset features: {list(ds.features.keys())}")

    # Auto-detect column names
    features = list(ds.features.keys())
    question_col = _find_column(features, ["question", "problem", "input", "prompt"])
    answer_col = _find_column(features, ["answer", "solution", "output", "target"])

    if question_col is None or answer_col is None:
        logger.warning(
            f"Could not auto-detect columns. Features: {features}. "
            f"Falling back to first two columns."
        )
        question_col = features[0]
        answer_col = features[1]

    logger.info(f"Using columns: question='{question_col}', answer='{answer_col}'")

    problems = []
    limit = max_problems if max_problems else len(ds)
    for idx in range(min(limit, len(ds))):
        row = ds[idx]
        problems.append({
            "problem_id": idx,
            "question": str(row[question_col]),
            "answer": _extract_boxed_answer(str(row[answer_col])),
            "raw": dict(row),
        })

    logger.info(f"Prepared {len(problems)} problems for inference")
    return problems


def _find_column(features: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    features_lower = {f.lower(): f for f in features}
    for c in candidates:
        if c.lower() in features_lower:
            return features_lower[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Answer Extraction & Grading
# ---------------------------------------------------------------------------

def _extract_boxed_answer(text: str) -> str:
    """
    Extract the answer from \\boxed{...} in a LaTeX-style solution.
    Handles nested braces. If no \\boxed found, return the full text stripped.
    """
    # Find all \boxed{...} occurrences, take the last one
    # (solutions typically have the final answer in the last \boxed)
    matches = list(_find_boxed_matches(text))
    if matches:
        return matches[-1].strip()
    # Fallback: look for "The answer is X" pattern
    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)", text)
    if m:
        return m.group(1).strip()
    # Last resort: return the last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


def _find_boxed_matches(text: str) -> List[str]:
    """Find all \\boxed{...} content, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        # Find the matching closing brace
        depth = 0
        start = idx + len("\\boxed{")
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            # No matching brace found
            i = start
    return results


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """
    Grade a model's predicted answer against the ground truth.

    Extracts boxed answers from the prediction, normalizes whitespace
    and common LaTeX formatting, then does string comparison.
    """
    pred_clean = _normalize_answer(_extract_boxed_answer(predicted))
    truth_clean = _normalize_answer(ground_truth)

    if not pred_clean or not truth_clean:
        return False

    return pred_clean == truth_clean


def _normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    s = text.strip()
    # Remove surrounding $ signs
    s = s.strip("$")
    # Remove \text{}, \mathrm{}, etc. wrappers
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)
    # Remove spaces
    s = re.sub(r"\s+", "", s)
    # Remove trailing period
    s = s.rstrip(".")
    # Lowercase
    s = s.lower()
    return s


# ---------------------------------------------------------------------------
# Prompt Formatting
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    """
    Format a LIMO question into a prompt suitable for math reasoning.

    We use a generic system/user format that works across models.
    The tokenizer's chat template (applied by nnsight/vllm) will handle
    the model-specific formatting.
    """
    return (
        "Solve the following math problem step by step. "
        "Show your reasoning and put your final answer in \\boxed{}.\n\n"
        f"Problem: {question}"
    )


def batch_problems(problems: List[Dict], batch_size: int) -> List[List[Dict]]:
    """Split problems into batches."""
    batches = []
    for i in range(0, len(problems), batch_size):
        batches.append(problems[i : i + batch_size])
    return batches
