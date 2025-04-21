from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import matplotlib as mpl

from analysis.analysis_helper import get_modified_performance, get_modified_feature_time, get_modified_task_time

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/"

sns.set_theme(style="whitegrid", palette="colorblind")

series = [
    get_modified_performance(path+"model_quality.csv").set_index(['feature_selector', 'ml_model', 'ml_task', 'fold'])['model_quality'],
    get_modified_feature_time(path+"feature_times.csv").groupby(['feature_selector', 'ml_model', 'ml_task', 'fold']).median()['feature_time'],
    get_modified_task_time(path+"task_times.csv").set_index(['feature_selector', 'ml_model', 'ml_task', 'fold'])['task_time'].map(lambda x: x.total_seconds())
]

df = pd.concat(series, axis=1)

print(df)
df = df.groupby(level=['feature_selector', 'ml_model', 'ml_task',]).median()
df = df.reset_index()
df['model_quality'] = df['model_quality'].clip(-1,1)
print(df)


plot = sns.relplot(df, x="feature_time", y="task_time", hue='feature_selector', size="model_quality", sizes=(10, 200),  col="ml_task", col_wrap=2)
plot.set(xscale="log", yscale="log")
plot.set(ylabel="Training Time", xlabel="Feature Time")
plot.set_titles(col_template="{col_name}")
plot.tight_layout()
#plt.show()
plot.savefig("out/rq2_tradeoff_times.pdf")

plt.clf()

str_trans = lambda x: x.replace('_', ' ').title()
plot = (so.Plot(df, y="model_quality", color="feature_selector")
        .pair(x=['feature_time', "task_time"]).facet(row="ml_task")
        .add(so.Dot()).limit(y=(-1,1))
        .layout(size=(15,20)).scale(x="log")
        .label(x=str_trans, y=str_trans, title=str_trans, legend="Feature Selector"))
#todo ggf. anpassen mit relativ-wert zu median
plt.tight_layout()
#plot.show()
plot.save("out/rq2_tradeoff_qual_time.pdf")

plt.clf()

df_mod = df.groupby(['feature_selector', 'ml_task'])['model_quality'].median().to_dict()
print(df_mod)
df['med_qual'] = df[['feature_selector', 'ml_task']].apply(axis=1, func=lambda x: df_mod[(x[0],x[1])])
df['rel_qual'] = df['model_quality'] - df['med_qual']
print(df)

figure = plt.figure(figsize=(15,20))

plot = (so.Plot(df, y="rel_qual", color="feature_selector")
        .on(figure)
        .pair(x=['feature_time', "task_time"]).facet(row="ml_task")
        .add(so.Dot()).limit(y=(-1,1))
        .layout(size=(15,20)).scale(x="log")
        .label(x=str_trans, y="Relative Model Quality", title=str_trans, legend="Feature Selector")).plot()
#todo ggf. anpassen mit relativ-wert zu median

for ax in figure.get_axes():
    ax.axhline(y=0, color="r", linestyle="--")

figure.tight_layout()
#figure.show()
figure.savefig("out/rq2_tradeoff_qual_time_rel.pdf")