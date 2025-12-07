import math
import numpy as np
import pandas as pd
from .exceptions import DataMismatchError

class FunctionMatcher:
    """
    Handles least-squares matching between training and ideal functions.
    """
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        # Basic sanity checks (use custom exception)
        if len(train_df) != len(ideal_df):
            raise DataMismatchError("Training and ideal data must have the same number of rows.")
        if not train_df["x"].equals(ideal_df["x"]):
            raise DataMismatchError("Training and ideal x-values are not aligned.")
        
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.best_matches: dict[str, tuple[str, float]] = {}
        self.max_devs: dict[str, float] = {}
        self.global_max_dev: float | None = None
        self.threshold: float | None = None

    @staticmethod
    def least_squares_error(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Return sum of squared errors between two aligned series."""
        diff = y_true - y_pred
        return float(np.sum(diff ** 2))

    def find_best_matches(self, train_cols=None) -> dict:
        """
        For each training column, find the ideal column with minimal SSE.
        Returns dict like: {'y1': ('y13', 34.08), ...}
        """
        if train_cols is None:
            train_cols = ["y1", "y2", "y3", "y4"]

        ideal_cols = [c for c in self.ideal_df.columns if c != "x"]
        best_matches = {}

        for t_col in train_cols:
            min_err = float("inf")
            best_col = None

            for i_col in ideal_cols:
                err = self.least_squares_error(self.train_df[t_col], self.ideal_df[i_col])
                if err < min_err:
                    min_err = err
                    best_col = i_col

            best_matches[t_col] = (best_col, min_err)

        self.best_matches = best_matches
        return best_matches

    def compute_deviations_and_threshold(self):
        """
        Compute max deviation per training-ideal pair, global max deviation
        and threshold = sqrt(2) * global_max_dev.
        """
        if not self.best_matches:
            raise RuntimeError("Call find_best_matches() before computing deviations.")
        
        max_devs = {}
        for t_col, (i_col, _) in self.best_matches.items():
            deviations = np.abs(self.train_df[t_col] - self.ideal_df[i_col])
            max_devs[t_col] = deviations.max()

        self.max_devs = max_devs
        self.global_max_dev = max(max_devs.values())
        self.threshold = math.sqrt(2) * self.global_max_dev

        return self.max_devs, self.global_max_dev, self.threshold
