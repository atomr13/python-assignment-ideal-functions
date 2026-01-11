from pathlib import Path
import pandas as pd
from .exceptions import MissingColumnError

"""
Dataset loader classes.

Loader classes for training, ideal and test datasets.
Loaders read a CSV into a pandas DataFrame and perform validation.
"""


# Class for loading CSV
class DatasetLoader:
    """
    Load a CSV dataset and keep it as a DataFrame.

    Args:
        path: Path to CSV file.
        name: Dataset name used in error messages.
        expected_columns: List of required column names.
    """

    def __init__(self, path: Path, name: str, expected_columns=None):
        self.path = Path(path)
        self.name = name
        self.expected_columns = expected_columns  # ✅ store expected columns
        self._df = None

    # CSV to Pandas
    def load(self) -> pd.DataFrame:
        """
        Load CSV file into a DataFrame.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            MissingColumnError: If expected columns are missing.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"{self.name} not found at: {self.path}")

        df = pd.read_csv(self.path)

        # Validate required columns (if provided)
        if self.expected_columns is not None:
            missing = [c for c in self.expected_columns if c not in df.columns]  # ✅ fixed attribute name
            if missing:
                raise MissingColumnError(
                    f"{self.name} is missing columns: {missing}"
                )

        self._df = df
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """
        Return the loaded DataFrame.

        Raises:
            RuntimeError: If .load() has not been called yet.
        """
        if self._df is None:
            raise RuntimeError(f"{self.name} has not been loaded yet. Call .load() first.")
        return self._df


# Class for Training Data
class TrainingDataLoad(DatasetLoader):
    """
    Loader for the training dataset.
    """
    def __init__(self, path: Path):
        super().__init__(
            path=path,
            name="Training data",
        )


# Class for Ideal Data
class IdealDataLoad(DatasetLoader):
    """
    Loader for the ideal functions dataset.
    """
    def __init__(self, path: Path):
        super().__init__(
            path=path,
            name="Ideal data",
        )


# Class for Test Data
class TestDataLoad(DatasetLoader):
    """
    Loader for the test dataset.
    """
    def __init__(self, path: Path):
        super().__init__(
            path=path,
            name="Test data",
        )
