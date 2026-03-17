from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.plots.plot_helper import add_median_labels


base = Path("/mnt/e/Uni/Thesis/fe4femo-feature_transformation/ml_analysis/out")
path = base / "rq2_preprocessing_dataset.csv"
out_path = base / "rq2_transformation_preprocessing_runtime.pdf"

TRANSFORMATION_LABELS = {
    "none": "None",
    "standardize": "Standardize",
    "minmax": "Min-Max",
    "signed-log1p": "Signed log1p",
    "quantile-normal": "Quantile Normalization",
    "yeo-johnson": "Yeo-Johnson",
    "pca": "PCA",
    "nystroem-rbf": "Nystroem RBF",
    "bin-ordinal": "Ordinal Binning",
}


sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv(path)
df["transformation_label"] = df["transformation"].map(TRANSFORMATION_LABELS).fillna(df["transformation"])

transformation_order = (
    df.groupby("transformation_label")["preprocessing_seconds"].median().sort_values().index.tolist()
)

plot = sns.catplot(
    df,
    x="preprocessing_seconds",
    y="transformation_label",
    estimator="median",
    errorbar="ci",
    kind="box",
    orient="h",
    legend="auto",
    height=8,
    order=transformation_order,
    medianprops={"linewidth": 2},
)
plot.set(
    xlabel="Preprocessing Runtime [s]",
    ylabel="Transformation",
)

for ax in plot.axes.flat:
    add_median_labels(ax, size="xx-small", fmt=2, scientific=False, boxen=False)

plot.tight_layout()
plot.savefig(out_path)
