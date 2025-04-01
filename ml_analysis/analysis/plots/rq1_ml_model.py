import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance

path = "/home/ubuntu/MA/data/extracted_ml_results/model_quality.csv"

sns.set_theme(style="whitegrid", palette="colorblind")


df = get_modified_performance(path)

d = sns.catplot(df, x="model_quality", y="ml_model", col="ml_task", col_wrap=2, estimator="median", errorbar="sd", kind="box", orient="h", facet_kws={"xlim":(-1,1)}, legend="auto")

#plt.savefig("test.pdf")
plt.show()