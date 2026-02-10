"""
Idea 1: Multilingual extraction prompt for Subtask 3.

Extends the engine's prompt with explicit multilingual instructions.
Used by ensemble_predictor; no edits to 11/1.
"""


def create_extraction_prompt_multilingual(syllogism: str) -> str:
    """
    Extraction prompt with Idea 1 (multilingual) instructions.
    The syllogism may be in any language; A/E/I/O and figure are universal.
    """
    return f"""You are a categorical logic expert. Extract the EXACT logical structure of this syllogism.

MULTILINGUAL: The syllogism may be in ANY language (English, Spanish, Dutch, Chinese, Bengali, Telugu, Swahili, Russian, etc.). The logical structure (A/E/I/O proposition types and syllogistic figure) is UNIVERSAL—extract it regardless of language. Ignore the concrete meanings of terms.

CONTENT-BLIND: To avoid content bias, treat each distinct term as an abstract placeholder. Focus ONLY on quantifier types and figure. Do NOT let plausibility or world knowledge affect your extraction.

NOTE: The entity tokens in the syllogism may be ENCRYPTED NONSENSE (e.g. a3f7b2c9)—they have NO semantic meaning. Ignore them and extract structure only.

SYLLOGISM TO ANALYZE:
"{syllogism}"

=== STEP 1: IDENTIFY THE THREE TERMS ===
Every categorical syllogism has exactly THREE terms:
- MAJOR TERM (P): The PREDICATE of the conclusion (what is said about the subject)
- MINOR TERM (S): The SUBJECT of the conclusion (what the conclusion is about)
- MIDDLE TERM (M): Appears in BOTH premises but NEVER in the conclusion (links the premises)

=== STEP 2: IDENTIFY PROPOSITION TYPES ===
For EACH statement (Premise1, Premise2, Conclusion), determine its type:

A (Universal Affirmative): "ALL X are Y", "Every X is Y", "Each X is Y", equivalent in any language
E (Universal Negative): "NO X is Y", "There are no X that are Y", "Nothing that is X is Y", equivalent in any language
I (Particular Affirmative): "SOME X are Y", "A few X are Y", "There exist X that are Y", equivalent in any language
O (Particular Negative): "SOME X are NOT Y", "Not all X are Y", equivalent in any language

=== STEP 3: DETERMINE THE FIGURE ===
The figure depends on the POSITION of the MIDDLE TERM in the premises:

FIGURE 1: M is PREDICATE in Premise1, SUBJECT in Premise2 (M-P, S-M)
FIGURE 2: M is PREDICATE in BOTH premises (P-M, S-M)
FIGURE 3: M is SUBJECT in BOTH premises (M-P, M-S)
FIGURE 4: M is SUBJECT in Premise1, PREDICATE in Premise2 (P-M, M-S)

=== RESPONSE FORMAT ===
Respond with ONLY this JSON (no other text):
{{
    "premise1_subject": "<subject of first premise>",
    "premise1_predicate": "<predicate of first premise>",
    "premise1_type": "<A|E|I|O>",
    "premise2_subject": "<subject of second premise>",
    "premise2_predicate": "<predicate of second premise>",
    "premise2_type": "<A|E|I|O>",
    "conclusion_subject": "<subject of conclusion = MINOR TERM>",
    "conclusion_predicate": "<predicate of conclusion = MAJOR TERM>",
    "conclusion_type": "<A|E|I|O>",
    "middle_term": "<term in both premises but not in conclusion>",
    "figure": <1|2|3|4>
}}

IMPORTANT: Do NOT assess validity. Only extract structure."""
