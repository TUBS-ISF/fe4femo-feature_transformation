import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance, get_replace_dictionary, get_order
from analysis.plots.plot_helper import add_median_labels

sns.set_theme(style="whitegrid", palette="colorblind")

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/"
df = pd.read_csv(path+"sel_stability.csv", index_col=[0,1], header=0)

df = df.reset_index()
df.replace(get_replace_dictionary(), inplace=True)
print(df)

#plot= sns.catplot(df, y="feature_selector", x="stability", col="ml_task",col_wrap=2, kind="bar", order=get_order())
#plot.set(xlim=(0,1), ylabel="Feature Selector", xlabel="Nogueira Feature Selection Stability")
#plot.set_titles(col_template="{col_name}")
#plot.savefig("out/rq1_stability_selector.pdf")
#plt.show()

plot = so.Plot(df, y="feature_selector", x="stability", xmin="lower", xmax="upper").facet(col="ml_task", wrap=2).add(so.Bar()).add(so.Range(linewidth=2)).scale(y=so.Nominal(order=get_order()))
plot = plot.layout(size=(8,12))
plot = plot.limit(xlim=(0,1))
plot = plot.label(x="Nogueira Feature Selection Stability", y="Feature Selector")

#plot.show()
plot.save("out/rq1_stability.pdf")
