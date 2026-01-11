import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from .exceptions import DataMismatchError


"""
Database utilities.
Write pandas dataframes into a SQLite database and read them back.
"""

#SQLite Operations
class DatabaseManager:
    def __init__(self, db_url: str = "sqlite:///assignment.db"):
        self.db_url = db_url
        self.engine: Engine = create_engine(db_url)

    def write_table(self, table_name: str, df: pd.DataFrame, if_exists: str = "replace"):
        """
        Write a dataframe to SQLite database.

        Args:
        :param self: Description
        :param table_name: Name of the target table
        :param df: dataframe to persist
        :param if_exists: if table exists check

        Raises:
            DataMismatchError: If df is None or empty.
        """
        if df is None or df.empty:
            raise DataMismatchError(
                f"DataFrame none or empty, can't write to table '{table_name}'"
            )

        df.to_sql(
            name=table_name,
            con=self.engine,
            if_exists=if_exists,
            index=False
        )

    def read_table(self, table_name) -> pd.DataFrame:
        """
        Read table from SQlite Databe into a Dataframe
        
        Args:
        :param self: Description
        :param table_name: Name of the table to read.
        
        Returns:
            Dataframe with the table contents. If reading fails, returns empty DataFrame.
        """
        try:
            df = pd.read_sql_table(
                table_name,
                con=self.engine
            )
        except Exception as e:
            print(f"[Error] Couldn't read from table '{table_name}': {e}")
            df = pd.DataFrame()

        return df
