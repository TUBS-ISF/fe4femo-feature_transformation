import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = "/home/ubuntu/MA/data"

df = pd.read_csv(data_path + "/runtime/sat.csv")

ax = plt.gca()

from cycler import cycler
colormap = plt.get_cmap("gist_ncar")
ax.set_prop_cycle(cycler('color', [colormap(i) for i in np.linspace(0, 0.9, 3)]))

cols = {"wallclockTimeS" : "kissat"}

min_val = df[cols.keys()].min().min()

for col, label in cols.items():
    y = df[col].sort_values().cumsum()
    y = np.array(y)
    y[y < min_val] = min_val
    x = [100 * float(i) / len(y) for i in range(len(y))]
    ax.step(y, x, label=label)

ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.set_xlabel("Cumulative Runtime [s]")
ax.set_ylabel("% Instances Solved or Timeout")
ax.set_xscale("log")
ax.legend()

plt.savefig("runtime_sat.pdf")