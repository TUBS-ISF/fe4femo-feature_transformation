import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from analysis.analysis_helper import get_modified_performance
from helper.load_dataset import load_dataset

path = "/home/ubuntu/MA/data/"
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
print(correlation)
obj = sns.clustermap(correlation, method="complete", cmap="RdBu", vmin=-1, vmax=1, figsize=(160, 160))

obj.savefig("test.pdf")
#plt.show()