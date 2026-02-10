<div align="center">

# 🌍 Instability-Triggered Compute for Multilingual Syllogistic Reasoning

### SemEval-2026 Task 11 — Subtasks 3 & 4

**Team lakksh**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![T5](https://img.shields.io/badge/T5--Small-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/google-t5/t5-small)
[![12 Languages](https://img.shields.io/badge/12_Languages-6_Families-green?style=for-the-badge)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

*Adaptive computation that spends effort only where content bias lurks — not everywhere.*

</div>

---

## 📌 Overview

Content bias in multilingual reasoning is **not uniform** — some inputs are stable, others are self-inconsistent. Uniformly anonymizing all inputs wastes compute and degrades accuracy through translation noise. This system treats instability as a **signal**, not a bug.

### Core Idea: Instability-Triggered Compute

```
                        Syllogism (any of 12 languages)
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Temperature Ensemble         │
                    │   τ = 0.2  →  v₀.₂            │
                    │   τ = 0.8  →  v₀.₈            │
                    └───────────────┬───────────────┘
                                    │
                          v₀.₂ == v₀.₈ ?
                          /              \
                        YES               NO
                        │                  │
                   ┌────▼────┐    ┌───────▼────────┐
                   │  RETURN  │    │  Translate →    │
                   │  early   │    │  Anonymize →    │
                   │  (save   │    │  Re-evaluate    │
                   │  compute)│    │  with T5        │
                   └─────────┘    └───────┬────────┘
                                          │
                                ┌─────────▼──────────┐
                                │   Hybrid Chooser    │
                                │   pick more stable  │
                                └────────────────────┘
```

**Why it works:** Content-driven reasoning is sensitive to temperature perturbations. Structurally grounded reasoning is not. Disagreement between `τ=0.2` and `τ=0.8` flags content interference.

---

## 🏆 Results

| Configuration | ACC (%) | TCE ↓ | Combined |
|---------------|---------|-------|----------|
| **Subtask 3** | | | |
| Gemini single-pass | 76.04 | 8.97 | 23.05 |
| Always anonymize | 48.44 | 30.89 | 10.86 |
| **Instability-triggered** | **90.62** | **10.42** | **26.38** |
| **Subtask 4** | | | |
| Filter Native, Validate Symbolic | 74.48 | 77.60 | 22.66 |

> **Key finding:** Adaptive compute outperforms both extremes. Always-anonymize *destroys* accuracy (48.44%). Single-pass ignores bias. Instability-triggered hits the sweet spot.

> **Negative result:** Subtask 4 TCE (77.60) vs Subtask 3 TCE (10.42) reveals that premise retrieval reintroduces content bias even when validation is content-free. The LLM favors semantically plausible premise pairs.

---

## 🌐 Supported Languages (6 Families)

| Family | Languages |
|--------|-----------|
| **Germanic** | German, Dutch |
| **Romance** | Spanish, French, Italian, Portuguese |
| **Slavic** | Russian |
| **Sino-Tibetan** | Chinese |
| **Bantu** | Swahili |
| **Indo-Aryan** | Bengali, Telugu |

---

## 📁 Repository Structure

```
Multilingual/
├── README.md
├── subtask_3/                          # Multilingual Validity Classification
│   ├── run_subtask3.py                 # Main orchestrator
│   ├── ensemble_predictor.py           # Instability-triggered compute logic
│   ├── prompts.py                      # Multilingual extraction prompt
│   ├── translator.py                   # Gemini-based translation (source → English)
│   ├── anonymizer.py                   # SHA-256 entity encryption
│   └── fill_missing_subtask3.py        # Backfill missing predictions
│
└── subtask_4/                          # Multilingual Retrieval + Validity
    ├── subtask4_engine.py              # Retrieval + symbolic validation pipeline
    └── utils.py                        # Multilingual sentence splitting
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install google-generativeai torch transformers tqdm scikit-learn spacy
```

### 2. Set API Key

```bash
export GEMINI_API_KEY='your-gemini-api-key'
```

### 3. Run Subtask 3 — Multilingual Validity

```bash
cd subtask_3

# Recommended: Instability-Triggered Compute (hybrid chooser)
python run_subtask3.py

# Alternative: Ensemble without hybrid chooser
python run_subtask3.py --no-hybrid

# Baseline: Single-pass (no ensemble)
python run_subtask3.py --no-ensemble
```

**Output:** `predictions.json` + `predictions.zip` (for Codabench upload)

### 4. Run Subtask 4 — Retrieval + Validity

```bash
cd subtask_4
python subtask4_engine.py --test_data <path_to_test_data> --workers 20
```

---

## 🔬 Key Design Decisions

### Why Not Always Anonymize?

| Strategy | ACC | TCE | Combined |
|----------|-----|-----|----------|
| Always anonymize | 48.44% | 30.89 | 10.86 |
| Instability-triggered | 90.62% | 10.42 | 26.38 |

Uniform anonymization introduces **translation noise** across all inputs. For stable predictions, this noise degrades accuracy without any bias benefit. Selective application saves ~40% compute and avoids unnecessary degradation.

### How Does Stability Detection Work?

```python
def is_stable(v_low, v_high):
    """Temperature-invariant agreement."""
    return v_low == v_high  # τ=0.2 vs τ=0.8
```

If both temperatures agree → **return immediately** (no translation, no anonymization).  
If they disagree → **trigger full pipeline**: translate → anonymize → re-evaluate.

### How Does Anonymization Work?

Entities are replaced with deterministic SHA-256 hashes (first 8 characters):

```
"Alle Katzen sind Säugetiere"  →  "Alle a3f7b2c9 sind e2d1c4b8"
```

The extraction prompt tells the model these tokens are meaningless nonsense, forcing structure-only extraction.

### Why Filter Native, Validate Symbolic?

**Subtask 4** decomposes reasoning across languages:
- **Native retrieval** — Premise selection in source language preserves word order, morphology, and discourse markers lost in translation.
- **English validation** — Retrieved premises are translated and validated using the content-free T5 Amnesiac parser.

This gives retrieval the benefit of multilingual understanding while insulating validity from content effects.

---

## ⚙️ System Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Translation | Gemini-2.0-Flash (τ=0.0) | Source → English pivot |
| Extraction | Gemini-2.0-Flash (τ=0.2, 0.8) | Structure parsing |
| Anonymization | SHA-256 hash | Deterministic entity encryption |
| Symbolic Parser | T5-Small (Amnesiac) | English structure extraction |
| Symbolic Kernel | Rule-based lookup | 15 valid syllogistic forms |
| Retrieval | Gemini-2.0-Flash (JSON mode) | Native-language premise selection |

---

## 📝 Citation

```bibtex
@inproceedings{lakksh-2026-semeval-multilingual,
    title     = {Revisiting Content Bias in Multilingual Natural Language 
                 Inference: A Case Study in Syllogistic Reasoning},
    author    = {Team lakksh},
    booktitle = {Proceedings of the 20th International Workshop on 
                 Semantic Evaluation (SemEval-2026)},
    year      = {2026},
    publisher = {Association for Computational Linguistics}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Built for [SemEval-2026 Task 11](https://semeval.github.io/SemEval2026/) · Subtasks 3 & 4**

*Content interference is input-dependent. Treat it that way.*

</div>
