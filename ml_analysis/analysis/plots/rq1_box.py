import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance, get_order, get_replace_dictionary

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/model_quality.csv"


sns.set_theme(style="whitegrid", palette="colorblind")


df = get_modified_performance(path)
print(df)

rev_map = dict((v, k) for k, v in get_replace_dictionary().items())

for name, group in df.groupby("ml_task"):
    plt.figure()
    plot = sns.catplot(group, y="feature_selector", x="model_quality", hue="ml_model", orient="h", estimator="median", kind="box", height=8, aspect=0.7, dodge=True, order=get_order(), legend_out=True)
    plot.refline(x=0, color="r", linestyle="--")
    plot.set(xlim=(-1,1), ylabel="Feature Selector", xlabel="Model Quality")
    plot.set_titles(title=name)
    plot.legend.set_title("ML Model")
    plot.tight_layout()
    #plt.show()
    plot.savefig(f"out/rq1_box_{rev_map[name]}.pdf")