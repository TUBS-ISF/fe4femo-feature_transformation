from pathlib import Path

import pandas as pd
import seaborn as sns

from analysis.plots.plot_helper import add_transformation_labels


MODEL_GROUPS = [
    ("rq1_transformation_quality_by_model_row1.pdf", ["gradboostForest", "randomForest", "kNN"]),
    ("rq1_transformation_quality_by_model_row2_without_mlp.pdf", ["adaboost", "SVM"]),
]


base = Path("/mnt/e/Uni/Thesis/fe4femo-feature_transformation/ml_analysis/out")
path = base / "rq1_quality_dataset.csv"


sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv(path)
df = add_transformation_labels(df)

transformation_order = (
    df.groupby("transformation_label")["model_quality"].median().sort_values().index.tolist()
)


for filename, models in MODEL_GROUPS:
    subset = df[df["ml_model"].isin(models)].copy()
    out_path = base / filename

    plot = sns.catplot(
        subset,
        x="model_quality",
        y="transformation_label",
        col="ml_model",
        col_order=models,
        estimator="median",
        errorbar="sd",
        kind="box",
        orient="h",
        legend="auto",
        height=5.5,
        aspect=0.9,
        order=transformation_order,
        medianprops={"linewidth": 2},
        sharex=False,
    )
    plot.set(
        xlabel="Model Quality",
        ylabel="Transformation",
    )

    for ax in plot.axes.flat:
        ax.set_xlim(-5, 1)

    plot.tight_layout()
    plot.savefig(out_path)
