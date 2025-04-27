import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance, get_replace_dictionary, get_order, get_reduction
from analysis.plots.plot_helper import add_median_labels

sns.set_theme(style="whitegrid", palette="colorblind")

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/feature_active.csv"

df = get_reduction(path)
print(df)

plot= sns.catplot(df, y="ml_model", x="rel_count", col="ml_task",col_wrap=2, kind="box",)
plot.set(xlim=(0,1), ylabel="Ml Model", xlabel="Relative Size of Selected Feature Subset")
plot.set_titles(col_template="{col_name}")
for ax in plot.axes.flat:
    add_median_labels(ax, size='x-small')
plot.tight_layout()

plot.savefig("out/rq1_count_reduction_model.pdf")
#plt.show()


plt.figure()
plot= sns.catplot(df, y="feature_selector", x="rel_count", col="ml_task",col_wrap=2, kind="box", order=get_order())
plot.set(xlim=(0,1), ylabel="Feature Selector", xlabel="Relative Size of Selected Feature Subset")
plot.set_titles(col_template="{col_name}")
for ax in plot.axes.flat:
    add_median_labels(ax, size='x-small')
plot.tight_layout()

plot.savefig("out/rq1_count_reduction_selector.pdf")
#plt.show()