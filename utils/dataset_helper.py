from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def load_dataset(
    source,
    dataset=None,
    file_path=None,
    target_column=None,
):
    """
    Load a regression dataset.

    Parameters
    ----------
    source : {"sklearn", "csv"}
        Dataset source.

    dataset : dict, optional
        OpenML dataset specification. Supports either

            {
                "dataset_name": "...",
                "version": ...
            }

        or

            {
                "data_id": ...
            }

    file_path : str or Path, optional
        Path to a CSV file.

    target_column : str, optional
        Name of the target column in the CSV.
    """

    if source == "sklearn":

        if dataset is None:
            raise ValueError(
                "dataset must be provided for sklearn datasets."
            )

        # ============================================================
        # Load by OpenML data_id
        # ============================================================

        if "data_id" in dataset and dataset["data_id"] is not None:

            dataset_var = fetch_openml(
                data_id=dataset["data_id"],
                as_frame=True,
            )

        # ============================================================
        # Load by dataset name
        # ============================================================

        elif "dataset_name" in dataset:

            if dataset.get("version") is None:

                dataset_var = fetch_openml(
                    name=dataset["dataset_name"],
                    as_frame=True,
                )

            else:

                dataset_var = fetch_openml(
                    name=dataset["dataset_name"],
                    version=dataset["version"],
                    as_frame=True,
                )

        else:

            raise ValueError(
                "dataset must contain either 'data_id' or 'dataset_name'."
            )

        X = dataset_var.data.to_numpy(dtype=np.float64)
        y = dataset_var.target.to_numpy(dtype=np.float64)

        return X, y

    elif source == "csv":

        if file_path is None:
            raise ValueError(
                "file_path must be provided for CSV datasets."
            )

        if target_column is None:
            raise ValueError(
                "target_column must be provided for CSV datasets."
            )

        df = pd.read_csv(Path(file_path))

        X = df.drop(columns=[target_column]).to_numpy(dtype=np.float64)
        y = df[target_column].to_numpy(dtype=np.float64)

        return X, y

    else:

        raise ValueError(
            f"Unknown source: {source}"
        )