import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().resolve().parent.parent.parent


def get_root() -> Path:
    return ROOT


def read_json(fname: str | Path) -> list[OrderedDict] | OrderedDict:
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Any, fname: str | Path) -> None:
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
