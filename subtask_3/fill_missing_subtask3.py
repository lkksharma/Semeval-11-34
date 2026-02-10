#!/usr/bin/env python3
"""
Fill missing predictions for SemEval-2026 Task 11 Subtask 3.

This avoids re-running the whole test set: it predicts ONLY ids that are missing
from the current predictions.json, then merges and rezips.

Usage:
  cd SemEval/11/3
  export GEMINI_API_KEY='your-key'
  python fill_missing_subtask3.py              # ensemble (default)
  python fill_missing_subtask3.py --no-ensemble  # single-run
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENGINE_DIR = REPO_ROOT / "1"
TEST_DATA_PATH = REPO_ROOT / "2" / "task11" / "test_data" / "subtask 3" / "test_data_subtask_3.json"
PRED_PATH = SCRIPT_DIR / "predictions.json"
ZIP_PATH = SCRIPT_DIR / "predictions.zip"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ensemble", action="store_true", help="Use single-run instead of ensemble")
    ap.add_argument("--limit-missing", type=int, default=None, help="Only fill first N missing ids (debug)")
    args = ap.parse_args()

    if not TEST_DATA_PATH.exists():
        raise SystemExit(f"Test data not found: {TEST_DATA_PATH}")

    # Load full test set
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Load existing predictions (if present)
    if PRED_PATH.exists():
        with open(PRED_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing_map = {item["id"]: bool(item["validity"]) for item in existing if "id" in item and "validity" in item}
    test_map = {item["id"]: item for item in test_data}

    missing_ids = [tid for tid in test_map.keys() if tid not in existing_map]
    if args.limit_missing:
        missing_ids = missing_ids[: args.limit_missing]

    print(f"Existing predictions: {len(existing_map)}")
    print(f"Total test items:      {len(test_data)}")
    print(f"Missing to fill:       {len(missing_ids)}")

    if not missing_ids:
        print("Nothing to do. Rebuilding zip only.")
    else:
        # Prepare model
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise SystemExit("Install dependency: pip install google-generativeai") from e

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Import engine from 11/1 without editing it
        sys.path.insert(0, str(ENGINE_DIR))
        os.chdir(ENGINE_DIR)

        # Build missing items list
        missing_items = [test_map[mid] for mid in missing_ids]

        if args.no_ensemble:
            from symbolic_syllogism_engine import SymbolicSyllogismEngine

            engine = SymbolicSyllogismEngine(gemini_model=model, use_existential_import=True)
            new_results = engine.predict_batch(missing_items, show_progress=True, anonymize=False)
            new_pairs = {r["id"]: bool(r["validity"]) for r in new_results}
        else:
            # Ensemble voting (Idea 3)
            sys.path.insert(0, str(SCRIPT_DIR))  # allow importing from 11/3
            from ensemble_predictor import predict_batch_hybrid

            new_results = predict_batch_hybrid(
                missing_items,
                model,
                temperatures=[0.2, 0.8],
                show_progress=True,
                instability_threshold=2,
            )
            new_pairs = {r["id"]: bool(r["validity"]) for r in new_results}

        existing_map.update(new_pairs)

    # Write merged predictions in REQUIRED format/order
    merged = [{"id": tid, "validity": existing_map[tid]} for tid in test_map.keys() if tid in existing_map]

    if len(merged) != len(test_data):
        # Still incomplete – don't produce a misleading zip
        missing_after = [tid for tid in test_map.keys() if tid not in existing_map]
        raise SystemExit(f"Still missing {len(missing_after)} ids after fill. Example: {missing_after[:5]}")

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    import zipfile

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(PRED_PATH, "predictions.json")

    print(f"\nWrote full predictions: {PRED_PATH}")
    print(f"Created zip:            {ZIP_PATH}")


if __name__ == "__main__":
    main()

