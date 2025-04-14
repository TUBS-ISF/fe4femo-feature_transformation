import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from analysis.analysis_helper import get_modified_performance

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/model_quality.csv"


sns.set_theme(style="whitegrid", palette="colorblind", rc={"figure.figsize":(6,15)}, font_scale=1)


df = get_modified_performance(path)
print(df)

plot = sns.catplot(df, y="feature_selector", x="model_quality", hue="ml_model", col="ml_task", col_wrap=2, orient="h", estimator="median", kind="box", height=10, aspect=0.7, dodge=True)
plot.refline(x=0, color="r", linestyle="--")
plot.set(xlim=(-1,1), ylabel="Feature Selector", xlabel="Model Quality")
plot.set_titles(col_template="{col_name}")
plot.tight_layout()

plt.show()
#plot.savefig("rq1_box.pdf")