"""Fixed label definitions for the Term Project (must not change).

Fruit (class) labels and Style labels exactly as specified in the assignment PDF.
These mappings are authoritative; the dataset folder structure is parsed onto them
in dataset.py.
"""

# Fruit (class) label -> canonical name
FRUIT_ID2NAME = {
    0: "apple",
    1: "asian pear",
    2: "banana",
    3: "cherry",
    4: "grape",
    5: "pineapple",
}

# Style label -> canonical name
STYLE_ID2NAME = {
    0: "pencil color",
    1: "oil painting",
    2: "water color",
}

NUM_FRUIT = len(FRUIT_ID2NAME)   # 6
NUM_STYLE = len(STYLE_ID2NAME)   # 3

FRUIT_NAME2ID = {v: k for k, v in FRUIT_ID2NAME.items()}
STYLE_NAME2ID = {v: k for k, v in STYLE_ID2NAME.items()}

# Common alternate spellings / folder-name variants -> canonical id.
# Filled defensively; extend in Step 0 once the real folder names are confirmed.
_FRUIT_ALIASES = {
    "apple": 0,
    "asian pear": 1, "asianpear": 1, "asian_pear": 1, "pear": 1,
    "banana": 2,
    "cherry": 3,
    "grape": 4, "grapes": 4,
    "pineapple": 5,
}

_STYLE_ALIASES = {
    "pencil color": 0, "pencil_color": 0, "pencilcolor": 0, "pencil coloring": 0,
    "pencil": 0, "pencil_coloring": 0, "coloredpencil": 0, "colored pencil": 0,
    "oil painting": 1, "oil_painting": 1, "oilpainting": 1, "oil": 1,
    "water color": 2, "water_color": 2, "watercolor": 2, "water": 2,
}


def _norm(s: str) -> str:
    return s.strip().lower().replace("-", " ").replace("_", " ")


def fruit_name_to_id(name: str) -> int:
    """Map a (possibly messy) folder/file token to a fruit id, or raise KeyError."""
    return _FRUIT_ALIASES[_norm(name)]


def style_name_to_id(name: str) -> int:
    """Map a (possibly messy) folder/file token to a style id, or raise KeyError."""
    return _STYLE_ALIASES[_norm(name)]
