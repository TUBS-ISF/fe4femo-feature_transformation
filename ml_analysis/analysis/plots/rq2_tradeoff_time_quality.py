import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

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
plt.show()

plt.clf()

str_trans = lambda x: x.replace('_', ' ').title()
plot = (so.Plot(df, y="model_quality", color="feature_selector")
        .pair(x=['feature_time', "task_time"]).facet(row="ml_task")
        .add(so.Dot()).limit(y=(-1,1))
        .layout(size=(15,20)).scale(x="log")
        .label(x=str_trans, y=str_trans, title=str_trans, legend="Feature Selector"))
plt.tight_layout()
plot.show()
#plot.savefig("rq2_tradeoff_time_stability.pdf")
