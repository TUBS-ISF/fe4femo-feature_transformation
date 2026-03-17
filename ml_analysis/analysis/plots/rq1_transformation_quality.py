from pathlib import Path

import seaborn as sns
import pandas as pd

from analysis.plots.plot_helper import add_transformation_labels


base = Path("/mnt/e/Uni/Thesis/fe4femo-feature_transformation/ml_analysis/out")
path = base / "rq1_quality_dataset.csv"
out_path = base / "rq1_transformation_quality.pdf"


sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv(path)
df = add_transformation_labels(df)

transformation_order = (
    df.groupby("transformation_label")["model_quality"].median().sort_values().index.tolist()
)

plot = sns.catplot(
    df,
    x="model_quality",
    y="transformation_label",
    estimator="median",
    errorbar="sd",
    kind="box",
    orient="h",
    legend="auto",
    height=8,
    order=transformation_order,
    medianprops={"linewidth": 2},
)
plot.set(
    xlabel="Model Quality",
    ylabel="Transformation",
)

for ax in plot.axes.flat:
    ax.set_xlim(-5, 1)

plot.tight_layout()
plot.savefig(out_path)
