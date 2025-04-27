import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_task_time, get_order
from analysis.plots.plot_helper import add_median_labels

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/task_times.csv"

sns.set_theme(style="whitegrid", palette="colorblind")

df = get_modified_task_time(path)
print(df)

plot = sns.catplot(df, x="task_time", y="feature_selector",  estimator="median", errorbar="ci", kind="boxen", orient="h", legend="auto", height=8, line_kws={"linewidth": 2},  order=get_order())
plot.set(xlabel="Runtime of Singular Feature Selector Execution [s]", ylabel="Feature Selector", xscale='log')

for ax in plot.axes.flat:
    add_median_labels(ax, size='xx-small', fmt=1, scientific=False, boxen=True)

plot.tight_layout()

#plot = sns.barplot(df, x="feature_time", y="feature_selector", hue="ml_task", log_scale=True)

#plt.show()
plot.savefig("out/rq2_training_time.pdf")