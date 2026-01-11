import math
import numpy as np
import pandas as pd
from .exceptions import DataMismatchError


"""
Ideal Function selection using least squares.

Selects, for each training function, the ideal function with the lowest SSE.
Computes per-function thresholds based on the maximum deviation times sqrt2.
"""


# SSE Function
class FunctionMatch:
    """
    Find best fitting ideal functions and compute deviation thresholds.

    Args:
        train_df: DataFrame containing x and training columns
        ideal_df: Dataframe containing x and ideal columns
        best_matches: Mapping for best found.
    """
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):        
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.best_matches = {}
        self.max_devs = {}
        self.global_max_dev= None
        self.threshold = None
    

    #least squares error
    @staticmethod
    def least_squares_error(y_act: pd.Series, y_pred: pd.Series):
        """
        Compute SSE between two series.
        """
        dif = y_act - y_pred
        squared = dif ** 2
        total_err = np.sum(squared)
        return float(total_err)


    #find ideal column with lowest SSE. 
    def find_matches(self, train_cols=None):
        """
        Select the ideal function with lowest SSE for each training function.
        
        Args:
            train_cols: Optional list of training columns.
        Returns:
            Dict mapping  training column
        """
        if train_cols is None:
            train_cols = ["y1", "y2", "y3", "y4"]

        ideal_cols = [col for col in self.ideal_df.columns if col != "x"]
        best_matches = {}

        for train_col in train_cols:
            min_err = float("inf")
            best_pair = None

            for ideal_col in ideal_cols:
                sse = self.least_squares_error(self.train_df[train_col], self.ideal_df[ideal_col])
                if sse < min_err:
                    min_err = sse
                    best_pair = ideal_col

            best_matches[train_col] = (best_pair, min_err)

        self.best_matches = best_matches
        return best_matches

    # max deviation for each pair & threshold.
    def dev_and_thresh(self):
        """
        Compute maximum deviation and sqrt2 thresholds.

        Returns:
            max_dev_mp: Dict training_col -> max abs deviation (train vs chosen ideal)
            thresholds: Dict training_col -> max_dev * sqrt(2)
            ideal_threshold: Dict ideal_col -> threshold 
        """
        max_dev_mp = {}
        thresholds = {}
        ideal_threshold = {}

        
        for train_col, (ideal_col, _) in self.best_matches.items():
            dev_series = self.train_df[train_col] - self.ideal_df[ideal_col]
            abs_dev = np.abs(dev_series)
            max_dev = float(abs_dev.max())

            
            max_dev_mp[train_col] = max_dev
            thresholds[train_col] = max_dev * math.sqrt(2)
            ideal_threshold[ideal_col] = thresholds[train_col]

        self.max_devs = max_dev_mp
        self.thresholds = thresholds
        self.ideal_threshold = ideal_threshold

        return max_dev_mp, thresholds, ideal_threshold
