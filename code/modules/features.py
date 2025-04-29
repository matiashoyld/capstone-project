"""Atomic text‑ and option‑feature helpers."""

import pandas as pd
import numpy as np
import re

def extract_text_features(text):
    if not isinstance(text, str) or text == "":
        return {
            "word_count": 0,
            "char_count": 0,
            "avg_word_length": 0.0,
            "digit_count": 0,
            "special_char_count": 0,
            "mathematical_symbols": 0,
            "latex_expressions": 0,
        }

    words = re.findall(r"\b\w+\b", text.lower())
    math_symbols = set("+-*/=<>±≤≥≠≈∞∫∑∏√^÷×∆∇∂")

    return {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "digit_count": sum(c.isdigit() for c in text),
        "special_char_count": sum(not c.isalnum() and not c.isspace() for c in text),
        "mathematical_symbols": sum(text.count(sym) for sym in math_symbols),
        "latex_expressions": sum(len(re.findall(p, text)) for p in [
            r"\\[a-zA-Z]+", r"\$.*?\$", r"\\\(.*?\\\)", r"\\\[.*?\\\]",
        ]),
    }

def jaccard_similarity(str1, str2):
    if not isinstance(str1, str) or not isinstance(str2, str) or str1 == "" or str2 == "":
        return 0.0
    set1 = set(re.findall(r"\b\w+\b", str1.lower()))
    set2 = set(re.findall(r"\b\w+\b", str2.lower()))
    union = len(set1 | set2)
    return float(len(set1 & set2) / union) if union else 0.0

def calculate_option_features(row, option_cols):
    options = [str(row[c]) if pd.notna(row[c]) else "" for c in option_cols]
    valid = [o for o in options if o]
    if not valid:
        return {
            "jaccard_similarity_std": 0.0,
            "avg_option_length": 0.0,
            "avg_option_word_count": 0.0,
        }

    lengths = [len(o) for o in valid]
    words = [len(re.findall(r"\b\w+\b", o.lower())) for o in valid]
    sims = [jaccard_similarity(a, b) for i, a in enumerate(valid) for b in valid[i+1:]]
    return {
        "jaccard_similarity_std": float(np.std(sims)) if sims else 0.0,
        "avg_option_length": float(np.mean(lengths)),
        "avg_option_word_count": float(np.mean(words)),
    }
