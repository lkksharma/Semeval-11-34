"""
Translation helper for Subtask 3.

Idea: run everything on ENGLISH, because the parser and rule-based
fallbacks are designed for English-style quantifiers.
"""

import time
from typing import Optional


def translate_to_english(
    syllogism: str, model, max_retries: int = 2
) -> Optional[str]:
    """
    Translate a syllogism into clear English, preserving logical structure
    and quantifiers. If already in English, return it unchanged.
    """
    prompt = f"""You are a professional translator and logician.

TASK:
- Translate the following syllogism into CLEAR ENGLISH.
- PRESERVE the logical structure and quantifiers:
  * 'all', 'no', 'some', 'not all', 'there are no', 'there exist', etc.
- If the input is ALREADY in English, output it unchanged.
- Do NOT explain, just output the translated syllogism text.

SYLLOGISM:
"{syllogism}"

ENGLISH SYLLOGISM:"""

    for attempt in range(max_retries):
        try:
            import google.generativeai as genai

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            text = (response.text or "").strip()
            # Heuristic: accept reasonably long outputs
            if text and len(text) > 20:
                return text
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            continue
    return None

