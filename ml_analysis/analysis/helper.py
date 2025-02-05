from cloudpickle import cloudpickle

def get_pickle_dict(path, file):
    with open(path+file, "br") as f:
        return cloudpickle.load(f)