from pathlib import Path

import pandas as pd


CHUNK_SIZE = 1000


def load_config(config_path: Path) -> pd.DataFrame:
    df = pd.read_csv(config_path, sep=r"\s+", engine="python")
    df = df.rename(
        columns={
            "NO": "config_id",
            "FOLD_NO": "fold",
            "FEATURE": "feature_selector",
            "TASK": "ml_task",
            "MODEL": "ml_model",
            "MODEL_HPO": "model_hpo",
            "SELECTOR_HPO": "selector_hpo",
            "MULTI_OBJECTIVE": "multi_objective",
            "TRANSFORMATION": "transformation",
        }
    )
    return df


def load_sacct(sacct_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sacct_path, sep="|")
    df[["parent_job_id", "array_task_id"]] = df["JobID"].str.extract(r"^(\d+)_(\d+)$")
    df = df.dropna(subset=["parent_job_id", "array_task_id"]).copy()
    df["parent_job_id"] = df["parent_job_id"].astype(int)
    df["array_task_id"] = df["array_task_id"].astype(int)
    df["elapsed_seconds"] = pd.to_timedelta(df["Elapsed"]).dt.total_seconds()
    # RQ2 runtime statistics should reflect only successful end-to-end evaluations.
    df = df[df["State"] == "COMPLETED"].copy()
    return df


def attach_config_ids(df: pd.DataFrame) -> pd.DataFrame:
    ordered_parents = sorted(df["parent_job_id"].unique())
    offset_map = {job_id: i * CHUNK_SIZE for i, job_id in enumerate(ordered_parents)}
    df["config_offset"] = df["parent_job_id"].map(offset_map)
    df["config_id"] = df["config_offset"] + df["array_task_id"]
    return df


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    config_path = base / "config.txt"
    sacct_path = base / "out" / "sacct_export_rq2.csv"
    out_path = base / "out" / "rq2_runtime_dataset.csv"

    config_df = load_config(config_path)
    sacct_df = load_sacct(sacct_path)
    sacct_df = attach_config_ids(sacct_df)

    merged = sacct_df.merge(
        config_df[
            [
                "config_id",
                "NAME",
                "fold",
                "feature_selector",
                "ml_task",
                "ml_model",
                "HPO_ITS",
                "model_hpo",
                "selector_hpo",
                "multi_objective",
                "transformation",
            ]
        ],
        on="config_id",
        how="left",
        validate="many_to_one",
    )

    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows to {out_path}")
