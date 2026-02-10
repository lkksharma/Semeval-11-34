"""
Idea 2: Content anonymization for Subtask 3 (multilingual).

Encrypts entities with a deterministic formula before extraction to eliminate
content bias. The model is told the entities are nonsense.
"""

import hashlib
import json
import re
import time
from typing import Optional


def _encrypt_entity(term: str, length: int = 8) -> str:
    """Encryption formula: first N chars of SHA256 hash. Deterministic nonsense."""
    h = hashlib.sha256(term.strip().encode("utf-8")).hexdigest()
    return h[:length]


def anonymize_with_gemini(syllogism: str, model, max_retries: int = 2) -> Optional[str]:
    """
    1. Ask Gemini to identify the 3 distinct terms.
    2. Encrypt each term with SHA256 (first 8 chars).
    3. Replace terms in syllogism with encrypted nonsense.
    """
    prompt = f"""List the 3 distinct logical terms in this syllogism (the 3 entities that appear in subject/predicate positions).
Output ONLY valid JSON: {{"terms": ["term1", "term2", "term3"]}}
Order: first appearance in the text.
The syllogism may be in any language. Extract the actual terms as they appear.

Syllogism:
"{syllogism}"

JSON:"""

    for attempt in range(max_retries):
        try:
            import google.generativeai as genai
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            text = (response.text or "").strip()
            # Extract JSON
            match = re.search(r'\{[^{}]*"terms"[^{}]*\}', text)
            if match:
                data = json.loads(match.group())
                terms = data.get("terms", [])
                if len(terms) >= 3:
                    terms = [str(t).strip() for t in terms[:3]]
                    result = syllogism
                    for term in sorted(terms, key=len, reverse=True):
                        if term:
                            enc = _encrypt_entity(term)
                            result = result.replace(term, enc)
                    if result and len(result) > 20:
                        return result
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(0.5)
    return None
