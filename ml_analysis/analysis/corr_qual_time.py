import pandas as pd
import numpy as np
import scipy.stats as stats

from analysis.analysis_helper import get_modified_performance, get_modified_feature_time

path_qual = "/home/ubuntu/MA/data/extracted_ml_results/model_quality.csv"
path_time = "/home/ubuntu/MA/data/extracted_ml_results/feature_times.csv"

df_qual = get_modified_performance(path_qual)
df_qual.set_index(["ml_task", "feature_selector", "ml_model", "fold"], inplace=True)

df_time = get_modified_feature_time(path_time)
df_time.set_index(["ml_task", "feature_selector", "ml_model", "fold"], inplace=True)

df = pd.concat([df_qual, df_time], axis=1)
print(df)


tmp_list = []
index = []
for name, group in df.groupby(["feature_selector"]):
    y = group['feature_time']
    def statistic_two_sided(x):
        return stats.spearmanr(x, y, alternative='two-sided').statistic
    def statistic(x):
        return stats.spearmanr(x, y).statistic
    res_exact = stats.permutation_test((group['model_quality'],), statistic=statistic, permutation_type='pairings', random_state=4242424242)
    res_exact_two_sided = stats.permutation_test((group['model_quality'],), statistic=statistic_two_sided, permutation_type='pairings', random_state=4242424242)
    print(f"{name}")
    index.append(name)
    tmp_list.append({
        ('two_sided', 'val'): res_exact_two_sided.statistic,
        ('two_sided', 'p'): res_exact_two_sided.pvalue,
        ('normal', 'val'): res_exact.statistic,
        ('normal', 'p'): res_exact.pvalue,
    })

df_res = pd.DataFrame(tmp_list, index=index)
df_res.columns = pd.MultiIndex.from_tuples(df_res.columns, names=['alt', 'type'])

print(df_res)
df_res.to_csv("corr_qual_time.csv")