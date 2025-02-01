import pandas as pd
from cloudpickle import cloudpickle

path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/"
file = "runtime_backbone#NDFS#randomForest#True#True#50#0.pkl"
with open(path+file, "br") as f:
    dictonary = cloudpickle.load(f)
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