from pathlib import Path

import cloudpickle
import pandas as pd


def _task_name(task: dict) -> str:
    key = task.get("key")
    if isinstance(key, tuple):
        key = key[0]
    return str(key).split("-")[0]


def _task_duration_seconds(task: dict) -> float:
    total = 0.0
    for segment in task.get("startstops", []):
        total += float(segment["stop"]) - float(segment["start"])
    return total


def extract_preprocessing_rows(output_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for pkl_path in sorted(output_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as handle:
            payload = cloudpickle.load(handle)

        run_config = payload["run_config"]
        preprocessing_seconds = sum(
            _task_duration_seconds(task)
            for task in payload.get("task_stream", [])
            if _task_name(task) == "precompute_feature_selection"
        )

        rows.append(
            {
                "file_name": pkl_path.name,
                "run_name": run_config["name"],
                "ml_task": run_config["task"],
                "feature_selector": run_config["features"],
                "ml_model": run_config["model"],
                "transformation": run_config["transformation"],
                "fold": run_config["foldNo"],
                "model_hpo": run_config["modelHPO"],
                "selector_hpo": run_config["selectorHPO"],
                "multi_objective": run_config["multiObjective"],
                "preprocessing_seconds": preprocessing_seconds,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    output_dir = base / "out" / "main"
    out_path = base / "out" / "rq2_preprocessing_dataset.csv"

    df = extract_preprocessing_rows(output_dir)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
