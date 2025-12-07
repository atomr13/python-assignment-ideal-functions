import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from .exceptions import DataMismatchError


class DatabaseManager:
    """
    Simple wrapper around SQLAlchemy engine to handle SQLite operations.
    """
    def __init__(self, db_url: str = "sqlite:///assignment.db"):
        self.db_url = db_url
        self.engine: Engine = create_engine(db_url)

    def write_table(self, name: str, df: pd.DataFrame, if_exists: str = "replace"):
        """
        Write a DataFrame to the database.
        Raises DataMismatchError if df is empty or None.
        """
        if df is None or df.empty:
            raise DataMismatchError(f"Cannot write empty or None DataFrame to table '{name}'.")
        df.to_sql(name, con=self.engine, if_exists=if_exists, index=False)

    def read_table(self, name: str) -> pd.DataFrame:
        """Read a table from the database into a DataFrame."""
        return pd.read_sql_table(name, con=self.engine)
