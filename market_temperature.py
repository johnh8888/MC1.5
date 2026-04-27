from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

ALL_NUMBERS = list(range(1, 50))


def _omission_map(draws: Sequence[Sequence[int]]) -> Dict[int, float]:
    omission = {n: float(len(draws) + 1) for n in ALL_NUMBERS}
    for idx, draw in enumerate(draws):
        seen = set()
        for raw in draw:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 49:
                seen.add(n)
        for n in seen:
            omission[n] = min(omission[n], float(idx + 1))
    return omission


def compute_market_temperature(draws: Sequence[Sequence[int]]) -> Dict[str, float]:
    if not draws:
        return {"cold_ratio": 0.0, "zone_entropy": 0.0}
    counts = {n: 0 for n in ALL_NUMBERS}
    zones = [0.0] * 5
    for draw in draws:
        for raw in draw:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 49:
                counts[n] += 1
                zones[min(4, (n - 1) // 10)] += 1.0
    cold_ratio = sum(1 for n in ALL_NUMBERS if counts[n] == 0) / 49.0
    total = float(sum(zones)) or 1.0
    probs = [z / total for z in zones if z > 0]
    zone_entropy = -sum(p * float(np.log(p + 1e-12)) for p in probs)
    zone_entropy = zone_entropy / float(np.log(5.0)) if zone_entropy > 0 else 0.0
    return {"cold_ratio": float(cold_ratio), "zone_entropy": float(zone_entropy)}
