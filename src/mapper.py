import pandas as pd
from .exceptions import DataMismatchError
from .db_manager import DatabaseManager

"""
Mapping of test points to selected ideal functions

This module assigns each test (x, y) point to the closest selected ideal function.
"""



#Maps test points to ideal functions
class Mapper:
    """
    Map test points to one of the selected ideal functions.

    Args:
        test_df: Dataframe with columns x and y for test points.
        ideal_df: DataFrame of ideal functions.
        best_matches: Dict mapping training col.
        ideal_threshold: Dict mapping ideal col.

    """
    def __init__(self, test_df: pd.DataFrame, ideal_df: pd.DataFrame, best_matches: dict, ideal_threshold: dict):
        self.test_df = test_df
        self.ideal_df = ideal_df
        self.best_matches = best_matches
        self.ideal_threshold = ideal_threshold
        self.selected_ideals = {}

        # pick best ideal function
        for test_col, match_info in best_matches.items():
            self.selected_ideals[test_col] = match_info[0]
        self.mapping_df = None
    
    # match test point to ideal function
    def map_test(self):
        """
        Map each test point to closest selected ideal function.

        Returns:
            DataFrame with columns: x, y, delta_y, ideal_function
        """
        map_rows = []

        for i, row in self.test_df.iterrows():
            x = row["x"]
            y = row["y"]

            ideal_row = self.ideal_df[self.ideal_df["x"] == x]
            if ideal_row.empty:
                continue

            clos_func = None
            small_dif = float("inf")

            for f_name in self.selected_ideals.values():
                try:
                    ideal_y = float(ideal_row[f_name].iloc[0])
                except(KeyError, IndexError):
                    continue
                
                dif = abs(y - ideal_y)

                if dif < small_dif:
                    small_dif = dif
                    clos_func = f_name

            if clos_func is not None:
                allowed = self.ideal_threshold.get(clos_func, None)
                if allowed is not None and small_dif <= allowed:
                    map_rows.append([x, y, small_dif, clos_func])


        self.mapping_df = pd.DataFrame(
            map_rows,
            columns=["x", "y", "delta_y", "ideal_function"]
        )
        return self.mapping_df

    # Use DB Manager to write mapping_df to database
    def write_to_db(self, db: DatabaseManager, table_name: str = "mapping"):
        """
        Persist the mapping results to the SQlite databes.
        """
        db.write_table(table_name, self.mapping_df)
