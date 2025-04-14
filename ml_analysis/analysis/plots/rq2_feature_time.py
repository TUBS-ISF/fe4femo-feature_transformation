import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_feature_time
from analysis.plots.plot_helper import add_median_labels

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/feature_times.csv"

sns.set_theme(style="whitegrid", palette="colorblind")

df = get_modified_feature_time(path)

plot = sns.catplot(df, x="feature_time", y="feature_selector",  estimator="median", errorbar="ci", kind="boxen", orient="h", legend="auto", height=8, line_kws={"linewidth": 2}, log_scale=True)
plot.set(xlabel="Cumulated Feature Computation Time", ylabel="Feature Selector", )

for ax in plot.axes.flat:
    add_median_labels(ax, size='x-small', boxen=True)

plot.tight_layout()

plt.show()
#plot.savefig("rq2_feature.pdf")