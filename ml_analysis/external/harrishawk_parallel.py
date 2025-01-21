import time
import warnings

import numpy as np
from distributed import worker_client
from zoofs import HarrisHawkOptimization


class HarrisHawkParallel(HarrisHawkOptimization):

    def _temporary_evaluate_fitness(self, model, x_train, y_train, x_valid, y_valid, target_individuals):
        _temp = self.individuals
        self.individuals = target_individuals
        res = self._evaluate_fitness(model, x_train, y_train, x_valid, y_valid)
        self.individuals = _temp
        return res

    @staticmethod
    def _negatable_objective(objective_function, model, x_train_copy, y_train, x_valid_copy, y_valid, chosen_features, kwargs, minimize: bool):
        X_masked = x_train_copy.iloc[:, chosen_features]
        score = objective_function(model, X_masked, y_train, X_masked, y_train, **kwargs)
        if not minimize:
            score = -score
        return score


    def _evaluate_fitness(self, model, x_train, y_train, x_valid, y_valid, particle_swarm_flag=0, dragon_fly_flag=0):
        future_scores = []

        with worker_client() as client:
            X = client.scatter(x_train)
            y = client.scatter(y_train)
            for individual in self.individuals:
                chosen_features = [index for index in range(x_train.shape[1]) if individual[index] == 1]

                feature_hash = "_*_".join(sorted(self.feature_list[chosen_features]))

                if feature_hash in self.feature_score_hash.keys():
                    score = self.feature_score_hash[feature_hash]
                else:
                    score = client.submit(self._negatable_objective, self.objective_function, model, X,
                                              y, X, y, chosen_features, self.kwargs, self.minimize)
                future_tuple = (feature_hash, individual, score)
                future_scores.append(future_tuple)
            scores = client.gather(future_scores)

        return_scores = []
        for feature_hash, individual, score in scores:
            self.feature_score_hash[feature_hash] = score
            if score < self.best_score:
                self.best_score = score
                self.best_dim = individual
                self.best_score_dimension = individual
            return_scores.append(score)
        return return_scores

    def fit(self, model, X_train, y_train, X_valid, y_valid, verbose=True):
        """
        Parameters
        ----------
        model: machine learning model's object
            The object to be used for fitting on train data

        X_train: pandas.core.frame.DataFrame of shape (n_samples, n_features)
            Training input samples to be used for machine learning model

        y_train: pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
            The target values (class labels in classification, real numbers in
            regression).

        X_valid: pandas.core.frame.DataFrame of shape (n_samples, n_features)
            Validation input samples

        y_valid: pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
            The target values (class labels in classification, real numbers in
            regression).

        verbose : bool,default=True
            Print results for iterations

        """
        #self._check_params(model, X_train, y_train, X_valid, y_valid)

        self.feature_score_hash = {}
        self.feature_list = np.array(list(X_train.columns))
        self.best_results_per_iteration = {}
        self.best_score = np.inf
        self.best_dim = np.ones(X_train.shape[1])

        self.initialize_population(X_train)

        if self.timeout is not None:
            timeout_upper_limit = time.time() + self.timeout
        else:
            timeout_upper_limit = time.time()
        for i in range(self.n_iteration):

            if (self.timeout is not None) & (time.time() > timeout_upper_limit):
                warnings.warn("Timeout occured")
                break

            # Logging warning if any entity in the population ends up having zero selected features
            self._check_individuals()

            self.fitness_scores = self._evaluate_fitness(model, X_train, y_train, X_valid, y_valid)

            self.gbest_individual = self.best_dim

            self.iteration_objective_score_monitor(i)

            self.e_0 = -1 + 2 * np.random.random(size=(self.population_size))
            self.e = 2 * self.e_0 * (1 - ((i + 1) / self.n_iteration))

            self.exploration_individuals_indexes = np.where(np.abs(self.e) >= 1)[0]
            self._exploration_phase()

            self.exploitation_individuals_indexes = np.where(np.abs(self.e) < 1)[0]
            self.r = np.random.random(len(self.exploitation_individuals_indexes))
            self.exploitation_energy = self.e[self.exploitation_individuals_indexes]

            self._soft_besiege()

            self._hard_besiege()

            self._soft_besiege_with_dives(model, X_train, y_train, X_valid, y_valid)

            self._hard_besiege_with_dives(model, X_train, y_train, X_valid, y_valid)

            self.verbose_results(verbose, i)

            self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list