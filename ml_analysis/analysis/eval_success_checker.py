from pathlib import Path

from analysis.analysis_helper import list_experiment_instances


def check_file_existence(file_path : Path) -> bool:
    return file_path.is_file()

if __name__ == '__main__':
    config_path = Path("~/fe4femo/ml_analysis/slurm_scripts/config.txt").expanduser()
    data_path = Path("~/fe4femo/ml_analysis/out/main/").expanduser()

    experiment_instances = list_experiment_instances(config_path, data_path)
    missing_instances = []
    for experiment_instance in experiment_instances:
        missing = False
        if not check_file_existence(experiment_instance.path_pickle):
            print(f"Missing pickle for {experiment_instance.name}")
            missing = True
        if (experiment_instance.is_model_hpo or experiment_instance.is_selector_hpo) and not check_file_existence(experiment_instance.path_journal):
            print(f"Missing journal for {experiment_instance.name}")
            missing = True
        if not check_file_existence(experiment_instance.path_dask):
            print(f"Missing dask for {experiment_instance.name}")
            missing = True
        if not check_file_existence(experiment_instance.path_log):
            print(f"Missing log for {experiment_instance.name}")
            missing = True

        if missing:
            missing_instances.append(experiment_instance.name)

    print("Missing instances:")
    print(missing_instances)