import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from analysis.analysis_helper import get_order
from helper.load_dataset import load_dataset

path = "/home/ubuntu/MA/raphael-dunkel-master/data/"
threshold = 0.8


#df = pd.read_csv(path, header=0, low_memory=False, index_col="modelNo")
#df.replace({False: 0, True: 1, None: pd.NA}, inplace=True)
df = load_dataset(path, "featureExtraction/values.csv")

variance_filter = VarianceThreshold()
variance_filter.set_output(transform="pandas")
df = variance_filter.fit_transform(df)

imputer = SimpleImputer(keep_empty_features=False, missing_values=pd.NA) #todo good?
imputer.set_output(transform="pandas")
df = imputer.fit_transform(df)

correlation =  df.corr(method="spearman")

name_mapper = lambda x: x.split("/", maxsplit=2)[2]
group_mapper = lambda x: x.split("/")[1]

features = list(correlation.index)
feature_groups = set(map(group_mapper, features))

group_pal = list(sns.color_palette("bright"))

group_lut = dict(SATZilla2024=(192.0/256, 192.0/256, 192.0/256), SATfeatPy=(105.0/256, 105.0/256, 105.0/256), FMBA=group_pal[6],
                 FM_Characterization=group_pal[1], DyMMer=group_pal[2], ESYES=group_pal[8])
dict(zip(feature_groups, group_pal))
group_colors = pd.Series(correlation.index, index=correlation.index.map(name_mapper)).apply(group_mapper).map(group_lut)


correlation.index = correlation.index.map(name_mapper)
correlation.rename(columns=name_mapper, inplace=True)

print(correlation)
plot = sns.clustermap(correlation, method="complete", cmap="RdBu", vmin=-1, vmax=1, figsize=(100, 100), col_cluster=True, row_cluster=True,
                      dendrogram_ratio=(.1, .2),
                      colors_ratio=.05,
                      cbar_pos=(.02, .32, .03, .2),
                      linewidths=0,
                      row_colors=group_colors, col_colors=group_colors,
                      tree_kws={"linewidths": 3,},
                      yticklabels=False,xticklabels=False
)
plot.ax_row_dendrogram.remove()
handles = [Patch(facecolor=group_lut[name]) for name in ["SATZilla2024", "SATfeatPy", "FMBA",  "FM_Characterization", "DyMMer", "ESYES"]]
plt.legend(handles, ["SATZilla", "SATfeatPy", "FMBA",  "FM Fact Label", "DyMMer", "ESYES"], title='Origin',
            loc='upper left', fontsize=70, bbox_to_anchor=(0.2, 3.35))

plot.ax_cbar.tick_params(labelsize=70)

#plot.ax_heatmap.set_xticklabels(plot.ax_heatmap.get_xmajorticklabels(), fontsize = 0.1)
#plot.ax_heatmap.set_yticklabels(plot.ax_heatmap.get_ymajorticklabels(), fontsize = 0.1)

plot.savefig("out/rq3_feature_corr.png")
#plot.savefig("test.png")
#plt.show()