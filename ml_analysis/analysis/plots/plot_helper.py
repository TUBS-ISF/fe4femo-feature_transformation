import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

import matplotlib.patheffects as path_effects


TRANSFORMATION_LABELS = {
    "none": "None",
    "standardize": "Standardize",
    "minmax": "Min-Max",
    "signed-log1p": "Signed log1p",
    "quantile-normal": "Quantile Normalization",
    "yeo-johnson": "Yeo-Johnson",
    "pca": "PCA",
    "nystroem-rbf": "Nystroem RBF",
    "bin-ordinal": "Ordinal Binning",
}


def add_transformation_labels(df: pd.DataFrame, source_col: str = "transformation",
                              target_col: str = "transformation_label") -> pd.DataFrame:
    df = df.copy()
    df[target_col] = df[source_col].map(TRANSFORMATION_LABELS).fillna(df[source_col])
    return df

# https://stackoverflow.com/a/63295846
def add_median_labels(ax: plt.Axes, fmt:int = 2, size='x-small', boxen=False, scientific=False) -> None:
    fmt = f".{fmt}{"E" if scientific else "f"}"
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 0 if boxen else 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       color='white', size=size)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground=median.get_color()),
            path_effects.Normal(),
        ])
