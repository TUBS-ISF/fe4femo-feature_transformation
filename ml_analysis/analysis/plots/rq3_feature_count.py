from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid", palette="colorblind")

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/feature_active.csv"

df = pd.read_csv(path, index_col=[0, 1, 2, 3, 4, 5, 6])
df = df[df.index.get_level_values(5) == False]
print(df)
df.reset_index(inplace=True)
df.replace({False: 0, True: 1}, inplace=True)
df.drop(columns=["ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"], inplace=True)
df = df.groupby(['ml_task', 'feature_selector']).sum()
print(df)
df.reset_index(inplace=True)
df = df.melt(id_vars=["ml_task", "feature_selector"], var_name="feature", value_name="count")
print(df)
df['group_rank'] = df.groupby(["ml_task", "feature_selector"])['count'].rank(method="dense", ascending=False)
print(df)

order = df.groupby(['feature'])['count'].sum().sort_values(ascending=False).index.values

#p = (
#    so.Plot(df, x="count", y="feature", color="feature_selector").add(so.Bars(), so.Agg(sum), so.Stack()).layout(size=(80,80)).scale(y=so.Nominal(order=order))
#)
#p.show()

plot = sns.catplot(df, x="count", y="feature", estimator=sum, orient="h", height=80, kind="bar", errorbar=None, order=order)
plot.tight_layout()
plt.show()
#plot.savefig("rq3_feature_count.pdf")
