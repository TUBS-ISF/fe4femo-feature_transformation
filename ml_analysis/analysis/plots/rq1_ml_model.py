import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance
from analysis.plots.plot_helper import add_median_labels

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/model_quality.csv"

sns.set_theme(style="whitegrid", palette="colorblind")


df = get_modified_performance(path)

plot = sns.catplot(df, x="model_quality", y="ml_model", col="ml_task", col_wrap=2, estimator="median", errorbar="sd", kind="box", orient="h", facet_kws={"xlim":(-1,1)}, legend="auto", medianprops={"linewidth": 2})
plot.refline(x=0, color="r", linestyle="--")
plot.set(xlim=(-1,1), ylabel="ML Model", xlabel="Model Quality")
plot.set_titles(col_template="{col_name}")
for ax in plot.axes.flat:
    add_median_labels(ax, size='small')
plot.tight_layout()

#plot.savefig("test.pdf")
plt.show()