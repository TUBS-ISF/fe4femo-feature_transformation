from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.plots.plot_helper import add_transformation_labels, add_median_labels


base = Path("/mnt/e/Uni/Thesis/fe4femo-feature_transformation/ml_analysis/out")
path = base / "rq2_runtime_relative_dataset.csv"
out_path_full = base / "rq2_transformation_runtime_relative_full.pdf"
out_path_clipped = base / "rq2_transformation_runtime_relative_clipped.pdf"


sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv(path)
df = add_transformation_labels(df)

transformation_order = (
    df.groupby("transformation_label")["delta_runtime"].median().sort_values().index.tolist()
)


def make_plot(clipped: bool, out_path: Path) -> None:
    plot = sns.catplot(
        df,
        x="delta_runtime",
        y="transformation_label",
        estimator="median",
        errorbar="ci",
        kind="box",
        orient="h",
        legend="auto",
        height=7,
        order=transformation_order,
        medianprops={"linewidth": 2},
    )
    plot.set(
        xlabel="Relative End-to-End Runtime vs. None [s] (Transformation - None)",
        ylabel="Transformation",
    )

    for ax in plot.axes.flat:
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        add_median_labels(ax, size="xx-small", fmt=2, scientific=False, boxen=False)
        if clipped:
            ax.set_xlim(-5, 15)

    plot.tight_layout()
    plot.savefig(out_path)
    plt.close(plot.figure)


make_plot(False, out_path_full)
make_plot(True, out_path_clipped)
