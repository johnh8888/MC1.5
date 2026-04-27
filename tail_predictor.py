from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Dict, List, Tuple

ALL_NUMBERS = list(range(1, 50))


def _normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    values = list(score_map.values())
    if not values:
        return score_map
    mn, mx = min(values), max(values)
    if mx == mn:
        return {k: 0.0 for k in score_map}
    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def _freq_map(draws: List[List[int]]) -> Dict[int, float]:
    freq = {n: 0.0 for n in ALL_NUMBERS}
    for draw in draws:
        for raw in draw:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 49:
                freq[n] += 1.0
    return freq


def _omission_map(draws: List[List[int]]) -> Dict[int, float]:
    omission = {n: float(len(draws) + 1) for n in ALL_NUMBERS}
    for idx, draw in enumerate(draws):
        for raw in draw:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 49 and omission[n] > float(idx + 1):
                omission[n] = float(idx + 1)
    return omission


def _load_draws(conn: sqlite3.Connection, window: int) -> List[List[int]]:
    rows = conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (int(window),),
    ).fetchall()
    return [json.loads(r["numbers_json"]) for r in rows]


def get_tail_scores(conn: sqlite3.Connection, window: int = 20) -> Dict[int, float]:
    draws = _load_draws(conn, window)
    if not draws:
        return {t: 0.0 for t in range(10)}

    freq = _normalize(_freq_map(draws))
    omission = _normalize(_omission_map(draws))

    scores = {t: 0.0 for t in range(10)}
    for n in ALL_NUMBERS:
        tail = n % 10
        scores[tail] += 0.65 * freq.get(n, 0.0) + 0.35 * omission.get(n, 0.0)

    return scores


def get_best_tail(conn: sqlite3.Connection) -> int:
    scores = get_tail_scores(conn)
    return max(scores.items(), key=lambda x: (x[1], -x[0]))[0] if scores else 0


def backtest_tail(conn: sqlite3.Connection, lookback: int = 120) -> Tuple[float, int, int]:
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC"
    ).fetchall()
    if len(rows) < 10:
        return 0.0, 0, 0

    start = max(10, len(rows) - int(lookback))
    hits = 0
    samples = 0
    miss_streak = 0
    max_miss = 0

    for i in range(start, len(rows)):
        history = [json.loads(r["numbers_json"]) for r in rows[max(0, i - 20):i]]
        if len(history) < 5:
            continue
        scores = get_tail_scores_from_history(history)
        best_tail = max(scores.items(), key=lambda x: (x[1], -x[0]))[0]
        actual_nums = json.loads(rows[i]["numbers_json"])
        actual_special = int(rows[i]["special_number"])
        winning_tails = {int(n) % 10 for n in actual_nums}
        winning_tails.add(actual_special % 10)
        hit = int(best_tail in winning_tails)
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
        else:
            miss_streak = 0

    hit_rate = hits / samples if samples else 0.0
    return float(hit_rate), int(samples), int(max_miss)


def get_tail_scores_from_history(draws: List[List[int]]) -> Dict[int, float]:
    freq = _normalize(_freq_map(draws))
    omission = _normalize(_omission_map(draws))
    scores = {t: 0.0 for t in range(10)}
    for n in ALL_NUMBERS:
        scores[n % 10] += 0.65 * freq.get(n, 0.0) + 0.35 * omission.get(n, 0.0)
    return scores
