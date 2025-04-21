from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from seaborn._core.groupby import GroupBy
from seaborn._core.moves import Move
from seaborn._core.typing import Default

@dataclass
class MiddleStack(Move):
    def _stack(self, df, orient):

        if df["baseline"].nunique() > 1:
            err = "Stack move cannot be used when baselines are already heterogeneous"
            raise RuntimeError(err)

        other = {"x": "y", "y": "x"}[orient]
        stacked_lengths = (df[other] - df["baseline"]).dropna().cumsum()
        offsets = stacked_lengths.shift(1).fillna(0)

        df[other] = stacked_lengths - (df[other]- df["baseline"])/2
        df["baseline"] = df["baseline"] + offsets

        return df

    def __call__(
        self, data: pd.DataFrame, groupby: GroupBy, orient: str, scales: dict[str],
    ) -> pd.DataFrame:

        groupers = ["col", "row", orient]
        return GroupBy(groupers).apply(data, self._stack, orient)

sns.set_theme(style="whitegrid", palette="colorblind")

path = "/home/ubuntu/MA/raphael-dunkel-master/data/extracted_ml_results/feature_active.csv"
path_groups = "/home/ubuntu/MA/raphael-dunkel-master/data/featureExtraction/groupMapping.csv"

groups = pd.read_csv(path_groups, index_col=0, header=0)['groupName']
group_count = groups.value_counts().to_dict()
groups = groups.to_dict()

df = pd.read_csv(path, index_col=[0, 1, 2, 3, 4, 5, 6])
df = df[df.index.get_level_values(5) == False]
print(df)
df.reset_index(inplace=True)
selector_count = df['feature_selector'].nunique()
df.replace({False: 0, True: 1}, inplace=True)
df.drop(columns=["ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"], inplace=True)
df['max_count'] = 1
df = df.groupby(['ml_task', 'feature_selector']).sum()
print(df)
df.reset_index(inplace=True)
df = df.melt(id_vars=["ml_task", "feature_selector", "max_count"], var_name="feature", value_name="count")
print(df)

#transform to groups
df['group'] = df['feature'].apply(lambda x: groups[x])
df.drop(columns=["feature"], inplace=True)
df = df.groupby(['ml_task', 'feature_selector', 'group']).sum().reset_index()
print(df)
df['rel_count'] =  df['count'] / df['max_count']
df['rel_count'] = df['rel_count'] / selector_count
print(df)
order = df.groupby(['group', 'feature_selector'])['rel_count'].mean().groupby(['group']).sum().sort_values(ascending=False).index.values
(
    so.Plot(df, y="group", x="rel_count", color="feature_selector", )
    #.facet(row="ml_task")
    .add(so.Bar(), so.Agg("mean"), so.Stack())
    #.add(so.Text(color="k"), MiddleStack())
    .layout(size=(10,14))
    .scale(y=so.Nominal(order=order))
).save("out/rq3_feature_group_count.pdf")