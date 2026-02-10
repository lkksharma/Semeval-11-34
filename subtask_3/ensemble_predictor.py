"""
Ensemble / Self-Consistency Voting for Subtask 3

Runs extraction at multiple temperatures and majority-votes on validity.
Idea 1: Multilingual prompt.
Idea 2: Optional content anonymization before extraction.

Hybrid mode:
- Run ORIGINAL ensemble first (higher ACC).
- Only if unstable/low-agreement, also run ANONYMIZED/ENCRYPTED ensemble (lower TCE).
- Choose the prediction with higher self-consistency.
"""

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Import from 11/1 (no edits to 11/1)
from symbolic_syllogism_engine import (
    check_validity_symbolic,
    parse_llm_response,
    rule_based_extract,
)

# Idea 1: Multilingual prompt (11/3 only)
from prompts import create_extraction_prompt_multilingual
from translator import translate_to_english


def extract_with_temperature(
    syllogism: str, model, temperature: float, max_retries: int = 2
) -> Optional[Dict]:
    """
    Extract logical structure using Gemini with a specific temperature.
    Uses multilingual prompt (Idea 1).
    """
    import google.generativeai as genai

    prompt = create_extraction_prompt_multilingual(syllogism)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=512,
                ),
            )
            parsed = parse_llm_response(response.text)

            if parsed and all(
                k in parsed
                for k in ["premise1_type", "premise2_type", "conclusion_type", "figure"]
            ):
                if (
                    parsed["premise1_type"] in "AEIO"
                    and parsed["premise2_type"] in "AEIO"
                    and parsed["conclusion_type"] in "AEIO"
                    and parsed["figure"] in [1, 2, 3, 4]
                ):
                    return parsed

            time.sleep(0.3)

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            continue

    return None


def predict_validity_ensemble(  # noqa: C901
    syllogism: str,
    model,
    temperatures: List[float] = (0.2, 0.8),
    use_existential_import: bool = True,
    use_fallback: bool = True,
    anonymize: bool = True,
) -> Tuple[bool, Dict]:
    """
    Predict validity via ensemble: run extraction at each temperature,
    get validity from symbolic kernel, majority vote.

    Idea 2: If anonymize=True, replace entities with ALPHA/BETA/GAMMA before extraction.
    """
    from anonymizer import anonymize_with_gemini

    text_to_parse = syllogism
    if anonymize:
        anonymized = anonymize_with_gemini(syllogism, model)
        if anonymized:
            text_to_parse = anonymized

    votes: List[Tuple[bool, float]] = []  # (validity, temperature)

    for temp in temperatures:
        structure = extract_with_temperature(text_to_parse, model, temp)
        if structure is None and use_fallback:
            structure = rule_based_extract(syllogism)  # fallback on original text

        if structure is not None:
            mood = (
                structure["premise1_type"],
                structure["premise2_type"],
                structure["conclusion_type"],
            )
            figure = structure["figure"]
            validity = check_validity_symbolic(mood, figure, use_existential_import)
            votes.append((validity, temp))

    if not votes:
        # All extractions failed - conservative default
        return False, {"error": "all_extractions_failed", "votes": []}

    validity_list = [v[0] for v in votes]
    validity_counts = Counter(validity_list)
    n_true, n_false = validity_counts[True], validity_counts[False]

    # Majority vote; on tie, prefer lowest-temperature (most deterministic) result
    if n_true > n_false:
        final_validity = True
    elif n_false > n_true:
        final_validity = False
    else:
        final_validity = votes[0][0]  # Tie: use lowest temp
    metadata = {
        "votes": validity_list,
        "n_agree": validity_counts[final_validity],
        "n_total": len(votes),
    }
    return final_validity, metadata


def predict_validity_hybrid(
    syllogism: str,
    model,
    temperatures: List[float] = (0.2, 0.8),
    use_existential_import: bool = True,
    use_fallback: bool = True,
    instability_threshold: int = 2,
) -> Tuple[bool, Dict]:
    """
    Hybrid chooser:
    1) Run ORIGINAL ensemble (anonymize=False).
    2) If original is fully stable (n_agree >= instability_threshold), return it.
    3) Else run ANONYMIZED/ENCRYPTED ensemble (anonymize=True) and pick the one
       with higher n_agree; on tie, prefer ORIGINAL.
    """
    # Step 0: translate to English (if needed)
    base_text = translate_to_english(syllogism, model) or syllogism

    v_orig, m_orig = predict_validity_ensemble(
        base_text,
        model,
        temperatures=temperatures,
        use_existential_import=use_existential_import,
        use_fallback=use_fallback,
        anonymize=False,
    )

    # If perfectly stable (e.g., 3/3 agree), don't pay extra API cost
    if m_orig.get("n_agree", 0) >= instability_threshold and m_orig.get("n_total", 0) >= instability_threshold:
        return v_orig, {**m_orig, "hybrid": "orig_only"}

    v_anon, m_anon = predict_validity_ensemble(
        base_text,
        model,
        temperatures=temperatures,
        use_existential_import=use_existential_import,
        use_fallback=use_fallback,
        anonymize=True,
    )

    score_orig = (m_orig.get("n_agree", 0), m_orig.get("n_total", 0))
    score_anon = (m_anon.get("n_agree", 0), m_anon.get("n_total", 0))

    if score_anon > score_orig:
        return v_anon, {**m_anon, "hybrid": "picked_anon", "orig_meta": m_orig}
    return v_orig, {**m_orig, "hybrid": "picked_orig", "anon_meta": m_anon}


def predict_batch_ensemble(
    syllogisms: List[Dict],
    model,
    temperatures: List[float] = (0.2, 0.8),
    show_progress: bool = True,
    anonymize: bool = True,
    **kwargs,
) -> List[Dict]:
    """Batch prediction with ensemble voting."""
    try:
        from tqdm import tqdm
        iterator = tqdm(syllogisms, desc="Ensemble predicting") if show_progress else syllogisms
    except ImportError:
        iterator = syllogisms

    results = []
    for item in iterator:
        validity, meta = predict_validity_ensemble(
            item["syllogism"],
            model,
            temperatures=temperatures,
            anonymize=anonymize,
            **kwargs,
        )
        results.append({"id": item["id"], "validity": validity, **meta})
    return results


def predict_batch_hybrid(
    syllogisms: List[Dict],
    model,
    temperatures: List[float] = (0.2, 0.8),
    show_progress: bool = True,
    instability_threshold: int = 2,
    **kwargs,
) -> List[Dict]:
    """Batch prediction with hybrid chooser (orig first, anon only if needed)."""
    try:
        from tqdm import tqdm

        iterator = (
            tqdm(syllogisms, desc="Hybrid predicting") if show_progress else syllogisms
        )
    except ImportError:
        iterator = syllogisms

    results: List[Dict] = []
    for item in iterator:
        validity, meta = predict_validity_hybrid(
            item["syllogism"],
            model,
            temperatures=temperatures,
            instability_threshold=instability_threshold,
            **kwargs,
        )
        results.append({"id": item["id"], "validity": validity, **meta})
    return results
