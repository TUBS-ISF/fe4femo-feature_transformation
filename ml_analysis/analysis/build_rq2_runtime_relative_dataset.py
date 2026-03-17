from pathlib import Path

import pandas as pd


GROUP_COLS = ["ml_task", "feature_selector", "ml_model", "fold"]
BASELINE_COL = "none"


def build_relative_dataset(df_matched: pd.DataFrame, value_prefix: str) -> pd.DataFrame:
    transformation_cols = [c for c in df_matched.columns if c not in GROUP_COLS]
    if BASELINE_COL not in transformation_cols:
        raise ValueError(f"Matched dataset does not contain baseline transformation '{BASELINE_COL}'")

    relative_rows = []
    for transformation in transformation_cols:
        if transformation == BASELINE_COL:
            continue

        rows = df_matched[GROUP_COLS].copy()
        rows["transformation"] = transformation
        rows["baseline_transformation"] = BASELINE_COL
        rows[f"{value_prefix}_value"] = df_matched[transformation]
        rows[f"baseline_{value_prefix}_value"] = df_matched[BASELINE_COL]
        rows[f"delta_{value_prefix}"] = rows[f"{value_prefix}_value"] - rows[f"baseline_{value_prefix}_value"]
        relative_rows.append(rows)

    return pd.concat(relative_rows, ignore_index=True)


def build_summary(df_relative: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    summary = (
        df_relative.groupby("transformation")[delta_col]
        .agg(["count", "median", "mean", "min", "max"])
        .sort_values("median")
        .reset_index()
    )
    return summary


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "out"
    in_path = base / "rq2_runtime_matched_dataset.csv"
    relative_path = base / "rq2_runtime_relative_dataset.csv"
    summary_path = base / "rq2_runtime_relative_summary.csv"

    df_matched = pd.read_csv(in_path)
    relative = build_relative_dataset(df_matched, value_prefix="runtime")
    summary = build_summary(relative, delta_col="delta_runtime")

    relative.to_csv(relative_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {len(relative)} relative rows to {relative_path}")
    print(f"Wrote {len(summary)} transformation summaries to {summary_path}")
