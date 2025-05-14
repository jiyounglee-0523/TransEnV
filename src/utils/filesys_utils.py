import json
import pickle
from typing import Any



def json_load(path: str) -> dict:
    """
    Load json file.

    Args:
        path (str): Path to the json file.

    Returns:
        dict: The object loaded from the json file.
    """
    with open(path, 'r') as f:
        return json.load(f)
    


def pickle_load(path: str) -> Any:
    """
    Load pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        Any: The object loaded from the pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f) 
    


def pickle_save(path: str, data: Any) -> None:
    """
    Save data to a pickle file.

    Args:
        path (str): Path to the pickle file.
        data (Any): Data to save.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)
