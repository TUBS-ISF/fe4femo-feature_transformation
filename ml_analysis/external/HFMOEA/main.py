from operator import itemgetter

import dask.distributed
from distributed import Variable
from joblib import Parallel, delayed

from external.HFMOEA.filter_methods import *
import time
import csv
import os
import math
import random
import matplotlib.pyplot as plt
from external.HFMOEA.utils import *
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator


def compute_sol(data : np.ndarray, target : np.ndarray, is_classification : bool, n_jobs : int = 1):
    functions = [MI, SCC, Relief, PCC, chi_square, info_gain, MAD, Dispersion_ratio, feature_selection_sim, Fisher_score]
    with worker_client() as client:
        data = client.scatter(data, direct=True)
        target = client.scatter(target, direct=True)
        sol_future = [ client.submit(fun, data, target, is_classification) for fun in functions]
        sol = client.gather(sol_future, direct=True)
    return pd.Series([x for x in sol if x is not None])

def compute(X_train_var, y_train_var, fold_vars : list, estimator, is_classification : bool, topk=25, pop_size=250, max_gen=1000, mutation_probability=0.06, n_jobs=1, sol = None, seed=42424242424242, dask_parallel : bool = False, verbose = False):
    rnd = np.random.default_rng(seed=seed)

    if sol is None:
        compute_sol(X_train_var.get(), y_train_var.get(), is_classification, n_jobs=n_jobs) # todo fix
    sol = sol.to_list()
    X_train = X_train_var.get().result().to_numpy()

    if pop_size < 10:
        pop_size = 10
        print("Population size cannot be less than 10.")

    init_size = len(sol)
    initial_chromosome = np.zeros(shape=(pop_size, X_train.shape[1]))
    for i in range(len(sol)):
        initial_chromosome[i, np.where(sol[i].ranks <= topk)[0]] = 1

    rand_size = pop_size - init_size
    rand_sol = rnd.integers(low=0, high=2, size=(rand_size, X_train.shape[1]))
    initial_chromosome[init_size:, :] = rand_sol

    # pop_shape = (pop_size,num_features)
    num_features = X_train.shape[1]
    num_mutations = (int)(pop_size * num_features * mutation_probability)
    solution = initial_chromosome
    gen_no = 1

    fun1_dict = {}
    while (gen_no <= max_gen):
        if verbose:
            print("Generation number: ", gen_no)

        # Generating offsprings
        solution2 = crossover(np.array(solution), offspring_size=(pop_size, num_features))
        solution2 = mutation(rnd, solution2, num_mutations=num_mutations)
        solution2 = check_sol(rnd, solution2)
        function1_values_new = function1(solution2, estimator, X_train_var, y_train_var, fold_vars, is_classification, n_jobs, dask_parallel, fun1_dict)
        function2_values_new = [function2(solution2[i]) for i in range(0, pop_size)]
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values_new[:], function2_values_new[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(function1_values_new[:], function2_values_new[:], non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(0, len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    function1_values = function1(np.array(solution), estimator, X_train_var, y_train_var, fold_vars, is_classification, n_jobs, dask_parallel, fun1_dict)
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]
    # Lets plot the final front now

    func1 = [i * -1 for i in function1_values]
    func2 = [j * -1 for j in function2_values]

    df = np.concatenate((np.expand_dims(np.asarray(func1), 1), np.expand_dims(np.asarray(func2), 1)), axis=1)

    pareto_index = is_pareto_efficient_indexed_reordered(df)
    pareto_front = [ (-1 * df[index][0], df[index][1], solution[index]) for index, isOptimal in enumerate(pareto_index) if isOptimal]
    return pareto_front

def reduceFeaturesMaxAcc(X_train_var, y_train_var, fold_vars : list, estimator, is_classification : bool, topk=25, pop_size=250, max_gen=1000, mutation_probability=0.06, n_jobs=1, sol=None, dask_parallel : bool = False, verbose = False):
    pareto_front = compute(X_train_var, y_train_var, fold_vars, estimator, is_classification, topk, pop_size, max_gen, mutation_probability, n_jobs, sol, dask_parallel=dask_parallel, verbose=verbose)
    acc, size, config = max(pareto_front, key=itemgetter(0))
    masked_x = X_train_var.get().result().loc[:, [ i == 1 for i in config]]
    return masked_x.columns.tolist()

