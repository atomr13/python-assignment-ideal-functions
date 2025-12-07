import pandas as pd
from .exceptions import DataMismatchError
from .db_manager import DatabaseManager

class TestMapper:
    """
    Maps test data points to the selected ideal functions
    using the deviation threshold.
    """
    def __init__(
        self,
        test_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        best_matches: dict,
        threshold: float
    ):
        if threshold is None:
            raise ValueError("Threshold must not be None.")
        if not best_matches:
            raise ValueError("best_matches must not be empty.")

        self.test_df = test_df
        self.ideal_df = ideal_df
        self.best_matches = best_matches
        self.threshold = threshold

        # Just store the chosen ideal functions (y13, y24, y36, y40, ...)
        self.selected_ideals = {
            t_col: i_col for t_col, (i_col, _) in best_matches.items()
        }

        self.mapping_df: pd.DataFrame | None = None

    def map_test_points(self) -> pd.DataFrame:
        """
        For each test (x, y), find the closest ideal function among the selected ones.
        Keep only those with deviation <= threshold.
        Returns a DataFrame with columns: x, y, delta_y, ideal_function.
        """
        results = []

        for _, row in self.test_df.iterrows():
            x_val = row["x"]
            y_val = row["y"]

            ideal_row = self.ideal_df[self.ideal_df["x"] == x_val]
            if ideal_row.empty:
                # no matching x in ideal table
                continue

            best_func = None
            min_dev = float("inf")

            for i_col in self.selected_ideals.values():
                ideal_y = float(ideal_row[i_col].iloc[0])
                deviation = abs(y_val - ideal_y)

                if deviation < min_dev:
                    min_dev = deviation
                    best_func = i_col

            if min_dev <= self.threshold:
                results.append([x_val, y_val, min_dev, best_func])

        self.mapping_df = pd.DataFrame(
            results,
            columns=["x", "y", "delta_y", "ideal_function"]
        )
        return self.mapping_df

    def write_to_db(self, db: DatabaseManager, table_name: str = "mapping"):
        """
        Write mapping_df to the database using DatabaseManager.
        """
        if self.mapping_df is None or self.mapping_df.empty:
            raise DataMismatchError("No mapping data to write. Run map_test_points() first.")
        db.write_table(table_name, self.mapping_df)
