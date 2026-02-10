#!/usr/bin/env python3
"""
SemEval-2026 Task 11, Subtask 3: Multilingual Syllogistic Reasoning

Uses the symbolic engine from 11/1 (imports only, no edits to 11/1).
By default uses Ensemble / Self-Consistency Voting (Idea 3).

Usage:
    cd SemEval/11/3
    export GEMINI_API_KEY='your-key'
    python run_subtask3.py              # HYBRID (recommended)
    python run_subtask3.py --no-ensemble   # single-run baseline

Output:
    - predictions.json
    - predictions.zip (for Codabench upload)
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENGINE_DIR = REPO_ROOT / "1"
TEST_DATA_PATH = REPO_ROOT / "2" / "task11" / "test_data" / "subtask 3" / "test_data_subtask_3.json"
OUTPUT_JSON = SCRIPT_DIR / "predictions.json"
OUTPUT_ZIP = SCRIPT_DIR / "predictions.zip"

# Import engine from 11/1 (no modifications to 11/1)
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))  # for prompts, anonymizer
os.chdir(ENGINE_DIR)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    ap.add_argument("--no-ensemble", action="store_true", help="Use single-run instead of ensemble")
    ap.add_argument("--no-hybrid", action="store_true", help="Use ensemble directly (no hybrid chooser)")
    args = ap.parse_args()

    print("=" * 70)
    print("SemEval-2026 Task 11 - Subtask 3: Multilingual Syllogistic Reasoning")
    print("=" * 70)

    if not TEST_DATA_PATH.exists():
        print(f"Error: Test data not found at {TEST_DATA_PATH}")
        sys.exit(1)

    with open(TEST_DATA_PATH) as f:
        test_data = json.load(f)
    if args.limit:
        test_data = test_data[: args.limit]
        print(f"Limited to {args.limit} samples")
    print(f"Loaded {len(test_data)} test samples")

    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("\nError: GEMINI_API_KEY or GOOGLE_API_KEY not set.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    except ImportError:
        print("\nError: pip install google-generativeai")
        sys.exit(1)

    if args.no_ensemble:
        from symbolic_syllogism_engine import SymbolicSyllogismEngine
        engine = SymbolicSyllogismEngine(gemini_model=gemini_model, use_existential_import=True)
        print("Engine: Symbolic (single-run)")
        results = engine.predict_batch(test_data, show_progress=True, anonymize=False)
    else:
        if args.no_hybrid:
            from ensemble_predictor import predict_batch_ensemble
            print("Engine: Ensemble (temps 0.2, 0.8) + multilingual prompt + encrypted anonymization")
            results = predict_batch_ensemble(
                test_data, gemini_model, temperatures=[0.2, 0.8], show_progress=True, anonymize=True
            )
        else:
            from ensemble_predictor import predict_batch_hybrid
            print("Engine: HYBRID chooser (orig first; anon only if unstable) + multilingual prompt")
            results = predict_batch_hybrid(
                test_data, gemini_model, temperatures=[0.2, 0.8], show_progress=True, instability_threshold=2
            )

    submission = [{"id": r["id"], "validity": r["validity"]} for r in results]

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"\nSaved {len(submission)} predictions to {OUTPUT_JSON}")

    import zipfile
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_JSON, "predictions.json")
    print(f"Created {OUTPUT_ZIP}")

    n_valid = sum(1 for p in submission if p["validity"])
    print(f"  Valid: {n_valid}, Invalid: {len(submission) - n_valid}")
    print("\nDone. Upload predictions.zip to Codabench.")


if __name__ == "__main__":
    main()
