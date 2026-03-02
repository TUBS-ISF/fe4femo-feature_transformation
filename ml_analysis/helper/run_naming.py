def create_run_name(
    features: str,
    task: str,
    model: str,
    modelHPO: bool,
    selectorHPO: bool,
    hpo_its: int,
    multi_objective: bool,
    foldNo: int,
) -> str:
    ret_value = f"{task}#{features}#{model}#{modelHPO}#{selectorHPO}"
    if modelHPO or selectorHPO:
        ret_value += f"#{hpo_its}"
    ret_value += f"#{multi_objective}"
    if foldNo >= 0:
        ret_value += f"#{foldNo}"
    return ret_value
