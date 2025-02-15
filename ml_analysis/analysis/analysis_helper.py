from cloudpickle import cloudpickle
from optuna import load_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def get_pickle_dict(file):
    with open(file, "br") as f:
        return cloudpickle.load(f)

def get_optuna_study(file, study_name):
    journal = JournalStorage(JournalFileBackend(file))
    study = load_study(storage=journal, study_name=study_name)
    return study, journal