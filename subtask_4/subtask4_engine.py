#!/usr/bin/env python3
"""
SemEval-2026 Task 11 Subtask 4: Multilingual Syllogistic Reasoning
Goal: Multilingual Retrieval + Classification
Approach:
1. Gemini 2.0 Flash for RETRIEVAL (Identify 2 relevant premises in Native Language).
2. Gemini 2.0 Flash for TRANSLATION (Translate subset to English).
3. SymbolicSyllogismEngine (from Subtask 2) for LOGIC VALIDATION (on English).
4. Gemini 2.0 Flash for NATIVE VALIDATION (Parallel check).
5. Ensemble Voting.
"""

import sys
import os
import json
import time
import argparse
import zipfile
import threading
import concurrent.futures
from tqdm import tqdm
import google.genai as genai
from google.genai import types

# --- Path Setup to import from Subtask 2 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
subtask2_dir = os.path.abspath(os.path.join(current_dir, '../2'))
sys.path.append(subtask2_dir)

try:
    from symbolic_syllogism_engine import SymbolicSyllogismEngine, check_validity_symbolic, extract_structure_with_gemini
    from utils import split_sentences_multilingual
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Please ensure {subtask2_dir} contains 'symbolic_syllogism_engine.py'")
    # Create valid dummy utils if import fails (to allow script to compile)
    def split_sentences_multilingual(text): return text.split(". ")

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")" 


if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")

# --- Prompts ---

def create_retrieval_prompt_native(sentences):
    """
    Asks the model to find the 2 premises relevant to the conclusion in the NATIVE language.
    The last sentence is assumed to be the conclusion BUT we should probably ask the model to identify it 
    or just follow the dataset convention (usually Logic contained in paragraph).
    Actually, Subtask 4 'irrelevant premises' implies a set of premises + conclusion.
    The 'syllogism' field in test data is a text paragraph.
    We'll assume the LAST sentence is the Conclusion, or ask the model to separate them.
    
    Better approach: Ask model to Output IDs of premises that entail the conclusion.
    """
    numbered_text = ""
    for i, p in enumerate(sentences):
        numbered_text += f"{i+1}. {p}\n"
        
    prompt = f"""You are a logician. Analysis the following text which contains a logical argument mixed with irrelevant sentences.
    
TEXT:
{numbered_text}

TASK:
1. Identify the Conclusion (usually the last sentence or indicated by "Therefore", "So", etc.).
2. Identify the TWO premises that logically link together to support (or attempt to support) that Conclusion.
3. Ignore irrelevant sentences.

OUTPUT JSON:
{{
    "conclusion_index": <int>,
    "relevant_premise_indices": [<int>, <int>]
}}
"""
    return prompt

def create_validation_prompt_native(premise1, premise2, conclusion):
    """
    Zero-shot validity check in Native Language.
    """
    prompt = f"""Assess the logical validity of this syllogism.
    
Premise 1: {premise1}
Premise 2: {premise2}
Conclusion: {conclusion}

Is this valid? (True/False).
If the conclusion necessarily follows from the premises, answer True.
If it is only 'likely' or 'plausible' but not logically guaranteed, answer False.

OUTPUT JSON:
{{
    "validity": <bool>
}}
"""
    return prompt

def create_translation_prompt(text):
    prompt = f"""Translate the following text to English efficiently. Preserve logical terms (All, Some, No, None) precisely.
    
Text: "{text}"
    
Output only the translated string.
"""
    return prompt

# --- Engine Class ---

class MultilingualReasoningEngine:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        
        # Initialize T5 Parser
        t5_path = os.path.join(subtask2_dir, 't5_syllogism_parser/checkpoint-1')
        t5_parser = None
        if os.path.exists(t5_path):
            try:
                from symbolic_syllogism_engine import T5SyllogismParser
                print(f"Loading T5 Parser from {t5_path}...")
                t5_parser = T5SyllogismParser(t5_path)
            except Exception as e:
                print(f"Failed to load T5 parser: {e}")
        else:
            print(f"T5 path not found: {t5_path}")

        # Init symbolic engine with T5
        self.symbolic_engine = SymbolicSyllogismEngine(
            gemini_model=self.client,
            t5_parser=t5_parser,
            use_existential_import=True,
            use_z3=False
        )

    def safe_generate(self, prompt, retries=3):
        """Safe generation with backoff."""
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"GenAI Error: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        # Fallback for plain text response
                        txt = response.text
                        import re
                        match = re.search(r'\{.*\}', txt, re.DOTALL)
                        if match:
                             return json.loads(match.group(0))
                    except:
                        pass
        return None

    def safe_translate(self, text):
        try:
             response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=create_translation_prompt(text)
            )
             return response.text.strip()
        except:
            return text

    def process_item(self, item):
        uid = item['id']
        text = item['syllogism']
        
        # 1. Split
        sentences = split_sentences_multilingual(text)
        if len(sentences) < 2:
            return {"id": uid, "validity": False, "relevant_premises": []}
            
        # 2. Retrieve (Native)
        retrieval_prompt = create_retrieval_prompt_native(sentences)
        ret_result = self.safe_generate(retrieval_prompt)
        
        if not ret_result:
            return {"id": uid, "validity": False, "relevant_premises": []}
            
        rel_indices = ret_result.get("relevant_premise_indices", [])
        
        # Convert to 0-based
        rel_indices_0b = [i if i < 10 else i-1 for i in rel_indices] # simple heuristic if model outputs 0 or 1 based
        # Actually better: check min value. If min is 1, assume 1-based.
        if rel_indices and min(rel_indices) >= 1:
             rel_indices_0b = [i-1 for i in rel_indices]
        else:
             rel_indices_0b = rel_indices
             
        # Sanity check bounds
        clean_indices = [i for i in rel_indices_0b if 0 <= i < len(sentences)]
        
        validity = False
        
        # 3. Validation Logic
        if len(clean_indices) >= 2:
            # Take top 2
            p1 = sentences[clean_indices[0]]
            p2 = sentences[clean_indices[1]]
            
            # Identify Conclusion (Last sentence usually)
            # Or from retrieval if available
            c_idx = ret_result.get("conclusion_index", -1) - 1
            if 0 <= c_idx < len(sentences) and c_idx not in clean_indices:
                conc = sentences[c_idx]
            else:
                # Heuristic: Find sentence NOT in premises
                remaining = [i for i in range(len(sentences)) if i not in clean_indices]
                if remaining:
                    conc = sentences[remaining[-1]] # Last non-premise sentence
                else:
                    conc = sentences[-1]
            
            # A. Symbolic Check (Translate -> T5 + Anonymization)
            triplet_text = f"{p1} {p2} Therefore, {conc}"
            eng_text = self.safe_translate(triplet_text)
            
            # predict_validity with anonymize=True (The Amnesiac Pipeline)
            validity, metadata = self.symbolic_engine.predict_validity(eng_text, anonymize=True)
            
            if not validity and metadata.get('error'):
                 # Fallback to Gemini Native if T5 failed critically
                 native_prompt = create_validation_prompt_native(p1, p2, conc)
                 native_res = self.safe_generate(native_prompt)
                 validity = native_res.get("validity", False) if native_res else False
        
        return {
            "id": uid,
            "validity": validity,
            "relevant_premises": clean_indices[:2] # Ensure strictly 2
        }

def run_subtask4(data_path, output_path, limit=None, max_workers=10):
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    if limit:
        data = data[:limit]
        
    # Thread-local engine initialization might be safer, but Client is thread-safe.
    # We will share the engine instance.
    engine = MultilingualReasoningEngine(API_KEY)
    results = []
    
    print(f"Processing {len(data)} items with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(engine.process_item, item): item for item in data}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(data)):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                item = future_to_item[future]
                print(f"Error processing {item['id']}: {e}")
                # Append fallback
                results.append({"id": item['id'], "validity": False, "relevant_premises": []})
        
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Zip
    zip_name = output_path.replace(".json", ".zip")
    with zipfile.ZipFile(zip_name, 'w') as zf:
        zf.write(output_path, arcname="predictions.json")
    print(f"Saved to {zip_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    
    run_subtask4(args.test_data, args.output, args.limit, args.workers)
