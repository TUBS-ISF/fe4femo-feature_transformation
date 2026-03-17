from pathlib import Path

import pandas as pd
import seaborn as sns


base = Path("/mnt/e/Uni/Thesis/fe4femo-feature_transformation/ml_analysis/out")
path = base / "rq1_quality_dataset.csv"
out_path = base / "rq1_transformation_quality_by_task.pdf"


sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv(path)

task_order = sorted(df["ml_task"].unique())
transformation_order = (
    df.groupby("transformation")["model_quality"].median().sort_values().index.tolist()
)

plot = sns.catplot(
    df,
    x="model_quality",
    y="transformation",
    col="ml_task",
    col_order=task_order,
    col_wrap=2,
    estimator="median",
    errorbar="sd",
    kind="box",
    orient="h",
    legend="auto",
    height=5,
    order=transformation_order,
    medianprops={"linewidth": 2},
    sharex=False,
)
plot.set(
    xlabel="Model Quality",
    ylabel="Transformation",
)

plot.tight_layout()
plot.savefig(out_path)
