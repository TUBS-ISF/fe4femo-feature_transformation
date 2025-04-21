import time

import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import plotly.express as px

def handleError(outpath):
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(outpath, format="pdf")
    time.sleep(2)

path_groups = "/home/ubuntu/MA/raphael-dunkel-master/data/featureExtraction/groupMapping.csv"
outpath = "out/feature_treemap.pdf"

handleError(outpath)

df = pd.read_csv(path_groups, index_col=0, header=0).reset_index()
df['origin'] = df['featureName'].apply(lambda x: x.split("/")[1])
df['groupName'] = df['groupName'].apply(lambda x: x.split("_")[-1])
group_count = df.groupby('origin').count()['featureName'].to_dict()
group_count['all'] = len(df.index)
print(group_count)
df = df.groupby(['origin', "groupName"]).count().reset_index().rename(columns={'featureName':"count"})
print(df)

fig = px.treemap(df, path=[px.Constant("all"), "origin", "groupName"], values="count", width=700, height=800,)
fig.update_traces(root_color="lightgrey")
fig.update_traces(textinfo="label+value", texttemplate="%{label}.<br>#F=%{value}", selector=dict(type="treemap"))
fig.update_traces(textposition="middle center")
fig.update_traces(marker=dict(cornerradius=5))
fig.update_layout(
    #uniformtext=dict(minsize=4, mode='hide'),
    margin = dict(t=5, l=5, r=5, b=5)
)
#fig.data[0].textinfo = 'label+value'

parent = fig.data[0]['parents']
child = fig.data[0]['labels']

for i,c in enumerate(child):
    if c in group_count:
        name = c
        if c == "FM_Characterization":
            name = "FM Fact Label"
        child[i] = f"{name}: {group_count[c]} Features"


#fig.show()
fig.write_image(outpath, format="pdf")