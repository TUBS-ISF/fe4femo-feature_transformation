import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance
from analysis.plots.plot_helper import add_median_labels


path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/feature_active.csv"
path_groups = "/home/ubuntu/MA/raphael-dunkel-master/data/featureExtraction/groupMapping.csv"


df = pd.read_csv(path, index_col=[0, 1, 2, 3, 4, 5, 6])
df = df[df.index.get_level_values(5) == False]
feature_max = len(df.columns)
print(df)
df.reset_index(inplace=True)
df.replace({False: 0, True: 1}, inplace=True)
df.drop(columns=["ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"], inplace=True)
df.set_index(['ml_task', 'feature_selector'], inplace=True)
df = df.sum(axis=1).reset_index()
df = df.rename(columns={df.columns[2]:"feature_count"})
print(df)
df['rel_count'] = df['feature_count'] / feature_max
print(df)

plot= sns.catplot(df, y="feature_selector", x="rel_count", col="ml_task",col_wrap=2, kind="box")
plot.set(xlim=(0,1), ylabel="Feature Selector", xlabel="Relative Size of Feature Set")
plot.set_titles(col_template="{col_name}")
for ax in plot.axes.flat:
    add_median_labels(ax, size='x-small')
plot.tight_layout()

#plot.savefig("test.pdf")
plt.show()