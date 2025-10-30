import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().resolve().parent.parent.parent


def get_root() -> Path:
    """
    Get project root directory.

    Returns
    -------
    Path
        Path to project root directory.
    """
    return ROOT


def read_json(fname: str | Path) -> list[OrderedDict] | OrderedDict:
    """
    Read JSON file with ordered dictionary preservation.

    Parameters
    ----------
    fname : str or Path
        Path to JSON file.

    Returns
    -------
    list[OrderedDict] or OrderedDict
        Loaded JSON content.
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Any, fname: str | Path) -> None:
    """
    Write content to JSON file with formatting.

    Parameters
    ----------
    content : Any
        Content to serialize to JSON.
    fname : str or Path
        Output file path.
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
