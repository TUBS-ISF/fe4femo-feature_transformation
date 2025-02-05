import pandas as pd

from analysis.helper import get_pickle_dict

path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/"
file = "runtime_backbone#HFMOEA#randomForest#True#True#50#0.pkl"
dictonary = get_pickle_dict(path, file)
tasks = []
for x in dictonary["task_stream"]:
    name = x["key"].split("-")[0]
    time = 0
    for y in x["startstops"]:
        time += y["stop"] - y["start"]
    tasks.append((name, time))

tasks = pd.DataFrame(tasks, columns=["name", "time"])
tasks["time"] = pd.to_timedelta(tasks["time"], unit="s")
cum_sum = tasks.groupby(['name']).sum()
print(cum_sum)

