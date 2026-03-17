from pathlib import Path

import cloudpickle
import pandas as pd
from sklearn.metrics import d2_absolute_error_score, matthews_corrcoef

from helper.feature_selection import set_njobs_if_possible
from helper.model_training import is_model_classifier


def compute_quality_from_pickle(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as handle:
        payload = cloudpickle.load(handle)

    run_config = payload["run_config"]
    y_test = payload["y_test"]

    qualities = []
    for trial_container in payload["trial_container"]:
        model = trial_container.model
        set_njobs_if_possible(model, 1)
        y_pred = model.predict(trial_container.x_test)
        if is_model_classifier(model):
            qualities.append(matthews_corrcoef(y_test, y_pred))
        else:
            qualities.append(float(d2_absolute_error_score(y_test, y_pred)))

    model_quality = max(qualities) if qualities else None

    return {
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
        "model_quality": model_quality,
        "trial_count": len(payload["trial_container"]),
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    output_dir = base / "out" / "main"
    out_path = base / "out" / "rq1_quality_dataset.csv"

    rows = [compute_quality_from_pickle(path) for path in sorted(output_dir.glob("*.pkl"))]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
