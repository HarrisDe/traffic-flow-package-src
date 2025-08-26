# energy_forecasting/deep_tf/horizons.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Union

def parse_horizons(horizons: Iterable[Union[int, str]]) -> Tuple[List[int], List[str]]:
    """
    Normalize user horizons to hourly steps + short string labels.

    Accepts any combination of:
      - ints:    1, 24, 168
      - strings: '1h','h','hour','24h','1d','d','day','168h','1w','w','week'

    Returns:
      steps:  [1, 24, ...]  (unique, order preserved)
      labels: ['1h','1d', ...] (same order as steps)
    """
    mapping = {
        "1h": 1, "h": 1, "hour": 1, "01h": 1,
        "24h": 24, "1d": 24, "d": 24, "day": 24,
        "168h": 168, "1w": 168, "w": 168, "week": 168,
    }
    norm = {1: "1h", 24: "1d", 168: "1w"}

    seen = set()
    steps: List[int] = []
    labels: List[str] = []

    for h in horizons:
        if isinstance(h, int):
            print(f"[isinstance int]key is {key}")
            if h not in (1, 24, 168):
                raise ValueError(f"Unsupported integer horizon {h}. Use 1, 24 or 168 for hourly data.")
            st = h
        else:
            key = str(h).strip().lower()
            print(f"[isinstance other] key is {key}")
            if key not in mapping:
                raise ValueError(f"Unsupported horizon string '{h}'. Use one of 1h/24h/168h or 1d/1w.")
            st = mapping[key]
        if st not in seen:
            seen.add(st)
            steps.append(st)
            labels.append(norm[st])
    return steps, labels