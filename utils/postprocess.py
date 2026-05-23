import re

import pandas as pd


def extract_answer_letter(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1).upper() if match else "?"


def build_submission(results: list, base_csv_path: str, save_path: str) -> None:
    submission = pd.read_csv(base_csv_path)
    submission["answer"] = results
    submission.to_csv(save_path, index=False)
    print(f"Saved submission to {save_path}")
