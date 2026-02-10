import re
import json
import os

def split_sentences_multilingual(text):
    """
    Robust sentence splitting for multilingual text.
    Handles common end-of-sentence punctuation across languages:
    - Standard: . ? !
    - Chinese/Japanese: 。 ？ ！
    - Bengali/Hindi: | (Danda)
    - Arabic: ؟
    """
    # Regex pattern explanation:
    # (?<=[...]) : Positive lookbehind to ensure we split AFTER the punctuation
    # [\.|\?|\!|\。|\？|\！|\||\؟] : The set of punctuation characters
    # \s+ : Followed by one or more whitespace
    # | (?<=[...]) $ : Or at the end of the string
    
    # We want to keep the punctuation with the sentence.
    # A simple split by punctuation often removes it.
    # Strategy: using re.split with capturing group keeps constraints, but lookbehind is cleaner if fixed width.
    # However, Python lookbehind must be fixed width.
    # Alternative: Replace <Punctuation><Space> with <Punctuation><SpecialSplitToken>
    
    # Map of all punctuation to handle
    puncts = ['.', '?', '!', '。', '？', '！', '|', '؟']
    
    # Protect common abbreviations? (Maybe overkill for this logic task, but good practice).
    # For now, we trust the clean nature of the dataset (usually full sentences).
    
    temp_text = text
    for p in puncts:
        # Replaces "Punctuation + Space" with "Punctuation + <SPLIT>"
        # Also handles "Punctuation + EndOfString"
        temp_text = temp_text.replace(f"{p} ", f"{p}<SPLIT>")
        temp_text = temp_text.replace(f"{p}  ", f"{p}<SPLIT>") # Handle double spaces
    
    # Special handle for end of string if not covered
    # Actually, simplistic replacement might fail on "Dr. Smith". 
    # But this is a syllogistic logic dataset, sentences are usually stand-alone statements.
    
    parts = temp_text.split("<SPLIT>")
    sentences = [s.strip() for s in parts if s.strip()]
    
    return sentences

def load_data(file_path):
    """Loads JSON data from the given path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_submission(data, output_file):
    """Saves the submission list to JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
