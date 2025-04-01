import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance, get_modified_feature_time, get_modified_task_time

path = "/home/ubuntu/MA/data/extracted_ml_results/"

sns.set_theme(style="whitegrid", palette="colorblind")


df = get_modified_performance(path+"model_quality.csv").merge(get_modified_feature_time(path+"feature_times.csv")).merge(get_modified_task_time(path+"task_times.csv"))
df['task_time'] = df['task_time'].map(lambda x: x.total_seconds())
df.set_index(['feature_selector', 'ml_model', 'ml_task', 'fold'], inplace=True)
print(df)
df = df.groupby(level=['feature_selector', 'ml_model', 'ml_task',]).median()
df = df.reset_index()
print(df)

#d = sns.catplot(df, x="model_quality", y="ml_model", col="ml_task", col_wrap=2, estimator="median", errorbar="sd", kind="box", orient="h", facet_kws={"xlim":(-1,1)}, legend="auto")
#g = sns.PairGrid(df, x_vars=['feature_time', "task_time"], y_vars=['model_quality'], hue='feature_selector', height=10)
#g.map(sns.scatterplot)
#g.set(xscale="log", ylim=(-1, 1))
#g.add_legend()

#plt.savefig("test.pdf")
#plt.show()

p = so.Plot(df, y="model_quality", color="feature_selector").pair(x=['feature_time', "task_time"]).facet(row="ml_task").add(so.Dot()).limit(y=(-1,1)).layout(size=(15,20)).scale(x="log")
p.save("rq2_tradeoff.pdf")
