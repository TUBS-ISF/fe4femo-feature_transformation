import numpy as np
from joblib import Parallel, delayed
from skrebate import MultiSURF
from skrebate.scoring_utils import MultiSURF_compute_scores

def find_neighbors(inst, datalen, distance_array):
    """ Identify nearest hits and misses within radius defined by average distance and standard deviation around each target training instance.
            This works the same regardless of endpoint type. """
    dist_vect = []
    for j in range(datalen):
        if inst != j:
            locator = [inst, j]
            if inst < j:
                locator.reverse()
            dist_vect.append(distance_array[locator[0]][locator[1]])

    dist_vect = np.array(dist_vect)
    inst_avg_dist = np.average(dist_vect)
    inst_std = np.std(dist_vect) / 2.
    # Defining a narrower radius based on the average instance distance minus the standard deviation of instance distances.
    near_threshold = inst_avg_dist - inst_std

    NN_near = []
    for j in range(datalen):
        if inst != j:
            locator = [inst, j]
            if inst < j:
                locator.reverse()
            if distance_array[locator[0]][locator[1]] < near_threshold:
                NN_near.append(j)

    return np.array(NN_near)


class MultiSURF_Parallel(MultiSURF):

    def _run_algorithm(self):
        """ Runs nearest neighbor (NN) identification and feature scoring to yield MultiSURF scores. """
        nan_entries = np.isnan(self._X)


        NNlist = Parallel(n_jobs=self.n_jobs)(delayed(find_neighbors)(datalen, self._datalen, self._distance_array) for datalen in range(self._datalen))

        if self.verbose:
            print("Finished NN identification")

        scores = np.sum(Parallel(n_jobs=self.n_jobs)(delayed(
            MultiSURF_compute_scores)(instance_num, self.attr, nan_entries, self._num_attributes, self.mcmap,
                                      NN_near, self._headers, self._class_type, self._X, self._y, self._labels_std, self.data_type)
            for instance_num, NN_near in zip(range(self._datalen), NNlist)), axis=0)

        return np.array(scores)