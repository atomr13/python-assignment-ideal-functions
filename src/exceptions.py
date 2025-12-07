class DataMismatchError(Exception):
    """Raised when datasets do not align or lengths differ."""
    pass


class MissingColumnError(Exception):
    """Raised when a required column is missing from a dataset."""
    pass
