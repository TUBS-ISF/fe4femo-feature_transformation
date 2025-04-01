import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_task_time

path = "/home/ubuntu/MA/data/extracted_ml_results/task_times.csv"

sns.set_theme(style="whitegrid", palette="colorblind")

df = get_modified_task_time(path)

d = sns.catplot(df, x="task_time", y="feature_selector", hue="ml_task", estimator="median", errorbar="sd", kind="bar", orient="h", legend="auto", height=15)
d.set(xscale="log")

#d = sns.barplot(df, x="feature_time", y="feature_selector", hue="ml_task", log_scale=True)

#plt.show()
plt.savefig("rq2_training.pdf")