from pathlib import Path

import pandas as pd
import seaborn as sns

from analysis.plots.plot_helper import add_transformation_labels


base = Path("/mnt/e/Uni/Thesis/fe4femo-feature_transformation/ml_analysis/out")
path = base / "rq1_quality_dataset.csv"
out_path_full = base / "rq1_transformation_quality_by_model_full.pdf"
out_path_clipped = base / "rq1_transformation_quality_by_model_clipped.pdf"


sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv(path)
df = add_transformation_labels(df)

model_order = (
    df.groupby("ml_model")["model_quality"].median().sort_values(ascending=False).index.tolist()
)
transformation_order = (
    df.groupby("transformation_label")["model_quality"].median().sort_values().index.tolist()
)


def make_plot(clipped: bool, out_path: Path) -> None:
    plot = sns.catplot(
        df,
        x="model_quality",
        y="transformation_label",
        col="ml_model",
        col_order=model_order,
        col_wrap=3,
        estimator="median",
        errorbar="sd",
        kind="box",
        orient="h",
        legend="auto",
        height=5,
        order=transformation_order,
        medianprops={"linewidth": 2},
        sharex=not clipped,
    )
    plot.set(
        xlabel="Model Quality",
        ylabel="Transformation",
    )

    if clipped:
        for ax in plot.axes.flat:
            ax.set_xlim(-5, 1)

    plot.tight_layout()
    plot.savefig(out_path)


make_plot(False, out_path_full)
make_plot(True, out_path_clipped)
