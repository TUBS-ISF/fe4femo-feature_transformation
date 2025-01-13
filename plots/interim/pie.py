import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

data_path = "/home/ubuntu/MA/data"

df = pd.read_csv(data_path + "/featureExtraction/groupMapping.csv")

ax = plt.gca()

df["sol"] = df["groupName"].apply(lambda x: x.split("_")[0])

counts = df["sol"].value_counts()
counts.rename(index={'FM':'FM Characterization'},inplace=True)
total = len(df.index)

ax.pie(counts, labels=counts.index, autopct=lambda p: '{:.0f}'.format(p * total / 100))
plt.tight_layout()
#plt.show()
plt.savefig("pie.pdf")