from cloudpickle import cloudpickle

def get_pickle_dict(file):
    with open(file, "br") as f:
        return cloudpickle.load(f)