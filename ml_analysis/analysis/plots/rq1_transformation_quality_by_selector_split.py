from pathlib import Path

import pandas as pd
import seaborn as sns

from analysis.plots.plot_helper import add_transformation_labels


SELECTOR_GROUPS = [
    ("rq1_transformation_quality_by_selector_pair_1.pdf", ["RFE", "mRMR"]),
    ("rq1_transformation_quality_by_selector_pair_2.pdf", ["embedded-tree", "all"]),
    ("rq1_transformation_quality_by_selector_pair_3.pdf", ["prefilter", "SATzilla"]),
    ("rq1_transformation_quality_by_selector_pair_4.pdf", ["kbest-mutalinfo", "SATfeatPy"]),
    ("rq1_transformation_quality_by_selector_pair_5.pdf", ["optuna-combined", "FM_Chara"]),
    ("rq1_transformation_quality_by_selector_pair_6.pdf", ["FMBA", "NDFS"]),
]


base = Path("/mnt/e/Uni/Thesis/fe4femo-feature_transformation/ml_analysis/out")
path = base / "rq1_quality_dataset.csv"


sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv(path)
df = add_transformation_labels(df)

transformation_order = (
    df.groupby("transformation_label")["model_quality"].median().sort_values().index.tolist()
)


for filename, selectors in SELECTOR_GROUPS:
    subset = df[df["feature_selector"].isin(selectors)].copy()
    out_path = base / filename

    plot = sns.catplot(
        subset,
        x="model_quality",
        y="transformation_label",
        col="feature_selector",
        col_order=selectors,
        estimator="median",
        errorbar="sd",
        kind="box",
        orient="h",
        legend="auto",
        height=5.5,
        aspect=1.0,
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
