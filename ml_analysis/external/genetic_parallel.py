import time
import warnings

import numpy as np
import scipy
from distributed import worker_client
from zoofs import GeneticOptimization



class GeneticParallel(GeneticOptimization):

    def __init__(
            self,
            objective_function,
            n_iteration: int = 1000,
            timeout: int = None,
            population_size=250,
            selective_pressure=2,
            elitism=2,
            mutation_rate=0.05,
            minimize=True,
            logger=None,
            seed=424242424242424424,
            **kwargs,
    ):
        super().__init__(objective_function, n_iteration, timeout, population_size, selective_pressure, elitism,
                         mutation_rate, minimize, logger, kwargs)
        self.rnd = np.random.default_rng(seed=seed)

    @staticmethod
    def _negatable_objective(objective_function, model, x_train_copy, y_train, X_test, y_test, chosen_features, kwargs, minimize: bool):
        X_train_masked = x_train_copy.iloc[:, chosen_features]
        X_test_masked = X_test.iloc[:, chosen_features]
        score = objective_function(model, X_train_masked, y_train, X_test_masked, y_test, **kwargs)
        if minimize:
            score = -score
        return score

    def _evaluate_fitness(self, model, x_train_var, y_train_var, x_valid_var, y_valid_var, particle_swarm_flag=0, dragon_fly_flag=0):
        future_scores = []

        with worker_client() as client:
            x_train = x_train_var.get()
            x_valid = x_valid_var.get()
            y_train = y_train_var.get()
            y_valid = y_valid_var.get()

            for individual in self.individuals:
                chosen_features = [index for index in range(x_train.result().shape[1]) if individual[index] == 1]

                feature_hash = "_*_".join(sorted(self.feature_list[chosen_features]))

                if feature_hash in self.feature_score_hash.keys():
                    score = self.feature_score_hash[feature_hash]
                else:
                    score = client.submit(self._negatable_objective, self.objective_function, model, x_train,
                                              y_train, x_valid, y_valid, chosen_features, self.kwargs, self.minimize, pure=False)
                future_tuple = (feature_hash, individual, score)
                future_scores.append(future_tuple)
            scores = client.gather(future_scores)

        return_scores = []
        for feature_hash, individual, score in scores:
            self.feature_score_hash[feature_hash] = score
            if score > self.best_score:
                self.best_score = score
                self.best_dim = individual
            return_scores.append(score)
        self.fitness_scores = return_scores

        ranks = scipy.stats.rankdata(return_scores, method='average')
        self.fitness_ranks = self.selective_pressure * ranks

    def initialize_population_rand(self, x):
        self.individuals = self.rnd.integers(0, 2, size=(self.population_size, x.shape[1]))

    def _select_individuals_rand(self, model, x_train, y_train, x_valid, y_valid):
        self._evaluate_fitness(model, x_train, y_train, x_valid, y_valid)

        sorted_individuals_fitness = sorted(
            zip(self.individuals, self.fitness_ranks), key=lambda x: x[1], reverse=True
        )
        elite_individuals = np.array(
            [individual for individual, fitness in sorted_individuals_fitness[: self.elitism]]
        )

        non_elite_individuals = np.array(
            [individual[0] for individual in sorted_individuals_fitness[self.elitism:]]
        )

        non_elite_individuals_fitness = [
            individual[1] for individual in sorted_individuals_fitness[self.elitism:]
        ]
        selection_probability = non_elite_individuals_fitness / np.sum(
            non_elite_individuals_fitness
        )

        selected_indices = self.rnd.choice(
            range(len(non_elite_individuals)), self.population_size // 2, p=selection_probability
        )
        selected_individuals = non_elite_individuals[selected_indices, :]
        self.fit_individuals = np.vstack((elite_individuals, selected_individuals))

    def _mutate_rand(self, array):
        mutated_array = np.copy(array)
        for idx, gene in enumerate(array):
            if self.rnd.random() < self.mutation_rate:
                array[idx] = 1 if gene == 0 else 0

        return mutated_array

    def _produce_next_generation_rand(self):
        new_population = np.empty(
            shape=(self.population_size, self.individuals.shape[1]), dtype=np.int32
        )
        for i in range(0, self.population_size, 2):
            parents = self.fit_individuals[
                self.rnd.choice(self.fit_individuals.shape[0], 2, replace=False), :
            ]
            crossover_index = self.rnd.integers(0, len(self.individuals[0]))
            new_population[i] = np.hstack(
                (parents[0][:crossover_index], parents[1][crossover_index:])
            )
            new_population[i + 1] = np.hstack(
                (parents[1][:crossover_index], parents[0][crossover_index:])
            )

            new_population[i] = self._mutate_rand(new_population[i])
            new_population[i + 1] = self._mutate_rand(new_population[i + 1])
        self.individuals = new_population

    def fit(self, model, X_train_var, y_train_var, X_valid_var, y_valid_var, verbose=True):
        """
        Parameters
        ----------
        model : machine learning model's object
           machine learning model's object

        X_train_var : pandas.core.frame.DataFrame of shape (n_samples, n_features)
           Training input samples to be used for machine learning model

        y_train_var : pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
           The target values (class labels in classification, real numbers in regression).

        X_valid_var : pandas.core.frame.DataFrame of shape (n_samples, n_features)
           Validation input samples

        y_valid_var : pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
            The target values (class labels in classification, real numbers in regression).

        verbose : bool,default=True
             Print results for iterations
        """
        #self._check_params(model, X_train, y_train, X_valid, y_valid)
        X_train = X_train_var.get().result()

        self.feature_score_hash = {}
        self.feature_list = np.array(list(X_train.columns))
        self.best_results_per_iteration = {}
        self.best_score = np.inf
        self.best_dim = np.ones(X_train.shape[1])

        self.initialize_population_rand(X_train)
        self.best_score = -1 * float(np.inf)
        self.best_scores = []

        if self.timeout is not None:
            timeout_upper_limit = time.time() + self.timeout
        else:
            timeout_upper_limit = time.time()
        for i in range(self.n_generations):

            if (self.timeout is not None) & (time.time() > timeout_upper_limit):
                warnings.warn("Timeout occured")
                break
            self._select_individuals_rand(model, X_train_var, y_train_var, X_valid_var, y_valid_var)
            self._produce_next_generation_rand()
            self.best_scores.append(self.best_score)

            self._iteration_objective_score_monitor(i)
            self._verbose_results(verbose, i)
            self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list