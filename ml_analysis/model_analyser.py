import sys
from pathlib import Path

import cloudpickle

from helper.load_dataset import is_task_classification

pathlist = Path(sys.argv[1]).rglob('*.pkl')
for path in pathlist:
    with open(path, 'rb') as f:
        instance = cloudpickle.load(f)
        run_config = instance["run_config"]
        is_classification = is_task_classification(run_config["task"])

