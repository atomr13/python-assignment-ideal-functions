class DataMismatchError(Exception):
    "Raised when there is a mismatch in data"
    pass
    

class MissingColumnError(Exception):
    """Raised when a required column is missing from a dataset."""
    pass
