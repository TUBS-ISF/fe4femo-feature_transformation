import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance

path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/model_quality.csv"

sns.set_theme(style="whitegrid", palette="colorblind")


df = get_modified_performance(path)

d = sns.catplot(df, x="model_quality", y="feature_selector", hue="ml_model", row="ml_task", estimator="median", errorbar="sd", kind="point", orient="h", dodge=.9, linestyles="none", facet_kws={"xlim":(-1,1)}, aspect=1, height=10, legend="auto")


plt.savefig("test.pdf")