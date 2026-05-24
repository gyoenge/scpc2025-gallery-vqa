import pandas as pd


def validate_qa(df: pd.DataFrame, balance: bool = False) -> pd.DataFrame:
    df = _filter_invalid_answers(df)
    df = filter_duplicates(df)
    _report_distribution(df)
    if balance:
        df = balance_answers(df)
    return df


def _filter_invalid_answers(df: pd.DataFrame) -> pd.DataFrame:
    valid = df["answer"].isin(["A", "B", "C", "D"])
    removed = (~valid).sum()
    if removed:
        print(f"Invalid answer filter: removed {removed} rows ({len(df)} → {valid.sum()})")
    return df[valid].reset_index(drop=True)


def filter_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["Question"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Duplicate filter: removed {removed} rows ({before} → {len(df)})")
    return df


def _report_distribution(df: pd.DataFrame) -> None:
    counts = df["answer"].value_counts().sort_index()
    total = len(df)
    parts = []
    for letter, count in counts.items():
        pct = count / total * 100
        flag = " ⚠" if pct < 15 else ""
        parts.append(f"{letter}: {count} ({pct:.1f}%){flag}")
    print("Answer distribution — " + " | ".join(parts))


def balance_answers(df: pd.DataFrame) -> pd.DataFrame:
    min_count = df["answer"].value_counts().min()
    df = (
        df.groupby("answer", group_keys=False)
        .apply(lambda x: x.sample(min_count, random_state=42))
        .reset_index(drop=True)
    )
    print(f"Balanced to {min_count} samples per class ({len(df)} total)")
    return df
