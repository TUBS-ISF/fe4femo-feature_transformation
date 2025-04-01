import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance

path = "/home/ubuntu/MA/data/extracted_ml_results/model_quality.csv"


sns.set_theme(style="whitegrid", palette="colorblind", rc={"figure.figsize":(6,15)}, font_scale=1)


df = get_modified_performance(path)
print(df)

for name, df_part in df.groupby("ml_task"):
    plt.clf()
    ax = sns.boxplot(data=df_part, y="feature_selector", x="model_quality", hue="ml_model", orient="h")
    ax.set_title(name)
    ax.set_xlabel("Model Quality")
    ax.set_ylabel("")
    ax.set_xlim(-1,1)
    plt.tight_layout()
    plt.savefig(f"{name}.pdf")