# modules/utils.py  ── stripped-down

import pandas as pd

def format_question_text(
    row: pd.Series,
    title_col: str = "question_title",
    option_cols: list[str] = ["option_a", "option_b", "option_c", "option_d", "option_e"],
    correct_option_col: str = "correct_option_letter",
) -> str:
    """
    Return a single string containing the question title plus its options.
    If the correct-option letter is available we label it; everything else is
    tagged as a wrong option.
    """
    title = str(row.get(title_col, "")) if pd.notna(row.get(title_col)) else ""
    out = [f"Question: {title}"]

    # map option letter (A/B/…) → column name (option_a/…)
    correct_col = None
    letter = str(row.get(correct_option_col, "")).strip().upper()
    if len(letter) == 1 and letter.isalpha():
        correct_col = f"option_{letter.lower()}"

    for col in option_cols:
        text = str(row.get(col, "")) if pd.notna(row.get(col)) else ""
        if not text:
            continue
        if col == correct_col:
            out.append(f"Correct Answer: {text}")
        else:
            out.append(f"Wrong Answer: {text}")

    return "\n".join(out)
