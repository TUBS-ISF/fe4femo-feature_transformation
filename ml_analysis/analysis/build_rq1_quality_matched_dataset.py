from pathlib import Path

import pandas as pd


GROUP_COLS = ["ml_task", "feature_selector", "ml_model", "fold"]
VALUE_COL = "model_quality"


def build_matched_dataset(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=GROUP_COLS,
        columns="transformation",
        values=VALUE_COL,
        aggfunc="first",
    )

    complete_groups = pivot.dropna(axis=0, how="any").copy()
    complete_groups.columns.name = None
    complete_groups = complete_groups.reset_index()
    return complete_groups


def build_summary(df_matched: pd.DataFrame) -> pd.DataFrame:
    transformation_cols = [c for c in df_matched.columns if c not in GROUP_COLS]
    summary = (
        df_matched[transformation_cols]
        .agg(["count", "median", "mean", "min", "max"])
        .T
        .sort_values("median", ascending=False)
    )
    summary.index.name = "transformation"
    return summary.reset_index()


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "out"
    in_path = base / "rq1_quality_dataset.csv"
    matched_path = base / "rq1_quality_matched_dataset.csv"
    summary_path = base / "rq1_quality_matched_summary.csv"

    df = pd.read_csv(in_path)
    matched = build_matched_dataset(df)
    summary = build_summary(matched)

    matched.to_csv(matched_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {len(matched)} matched groups to {matched_path}")
    print(f"Wrote {len(summary)} transformation summaries to {summary_path}")
