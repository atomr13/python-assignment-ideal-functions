from pathlib import Path
import pandas as pd
from .exceptions import MissingColumnError

class BaseDatasetLoader:
    """
    Base class for loading CSV datasets.
    Subclasses define expected_columns.
    """
    def __init__(self, path: Path, name: str, expected_columns=None):
        self.path = Path(path)
        self.name = name
        self.expected_columns = expected_columns or []
        self._df = None

    def load(self) -> pd.DataFrame:
        """Load the CSV file into a pandas DataFrame and validate columns."""
        if not self.path.exists():
            raise FileNotFoundError(f"{self.name} file not found at: {self.path}")

        df = pd.read_csv(self.path)

        # validate required columns if specified
        missing = [c for c in self.expected_columns if c not in df.columns]
        if missing:
            raise MissingColumnError(
                f"{self.name} is missing required columns: {missing}"
            )

        self._df = df
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError(f"{self.name} has not been loaded yet. Call .load() first.")
        return self._df


class TrainingDataLoader(BaseDatasetLoader):
    """Loader for training data (x + 4 training functions)."""
    def __init__(self, path: Path):
        super().__init__(
            path=path,
            name="Training data",
            expected_columns=["x", "y1", "y2", "y3", "y4"],
        )


class IdealDataLoader(BaseDatasetLoader):
    """Loader for ideal functions (x + 50 ideal functions)."""
    def __init__(self, path: Path):
        # only enforce that 'x' exists; y1..y50 are accepted as present
        super().__init__(
            path=path,
            name="Ideal data",
            expected_columns=["x"],
        )


class TestDataLoader(BaseDatasetLoader):
    """Loader for test data (x, y)."""
    def __init__(self, path: Path):
        super().__init__(
            path=path,
            name="Test data",
            expected_columns=["x", "y"],
        )
