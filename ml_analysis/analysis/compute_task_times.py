import pandas as pd

from analysis.analysis_helper import get_pickle_dict


def get_task_cumsum(file):
    dictonary = get_pickle_dict(file)
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
    return cum_sum

if __name__ == '__main__':
    path = "/home/ubuntu/MA/fe4femo/ml_analysis/copy/"
    instance = "runtime_backbone#HFMOEA#randomForest#True#True#50#0.pkl"
    print(get_task_cumsum(path + instance))