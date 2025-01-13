import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = "/home/ubuntu/MA/data"

df = pd.read_csv(data_path + "/runtime/sharpsat.csv")

ax = plt.gca()

from cycler import cycler
colormap = plt.get_cmap("gist_ncar")
ax.set_prop_cycle(cycler('color', [colormap(i) for i in np.linspace(0, 0.9, 3)]))

cols = {
    "approxmc" : "ApproxMC",
    "countantom" : "CountAntom",
    "d4v2_23" : "d4v2_23",
    "d4v2_24" : "d4v2_24",
    "exactmc_arjun" : "ExactMC_arjun",
    "ganak" : "ganak",
    "sharpsattd" : "SharpsatTD"
}

for col in cols.keys():
    wrong_index = df.loc[pd.isna(df[col+"_modelCount"]), :].index
    df.loc[wrong_index, col+"_wallclockTimeS"] = 3600

cols_mod = [col+"_wallclockTimeS" for col in cols.keys()]

min_val = df[cols_mod].min().min()

for col, label in cols.items():
    y = df[col+"_wallclockTimeS"].sort_values().cumsum()
    y = np.array(y)
    y[y < min_val] = min_val
    x = [100 * float(i) / len(y) for i in range(len(y))]
    ax.step(y, x, label=label)

ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.set_xlabel("Cumulative Runtime [s]")
ax.set_ylabel("% Instances Solved or Timeout")
ax.set_xscale("log")
ax.legend()

plt.savefig("runtime_sharpsat.pdf")