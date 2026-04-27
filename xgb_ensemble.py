from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Dict, List, Sequence, Tuple

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49],
    "蛇": [2, 14, 26, 38],
    "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40],
    "虎": [5, 17, 29, 41],
    "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43],
    "猪": [8, 20, 32, 44],
    "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46],
    "猴": [11, 23, 35, 47],
    "羊": [12, 24, 36, 48],
}


def get_zodiac_by_number(number: int) -> str:
    for zodiac, nums in ZODIAC_MAP.items():
        if int(number) in nums:
            return zodiac
    return "马"


def check_two_zodiac_strict(picks: Sequence[str], main_numbers: Sequence[int], special_number: int) -> bool:
    winning = {get_zodiac_by_number(int(n)) for n in main_numbers}
    winning.add(get_zodiac_by_number(int(special_number)))
    return all(p in winning for p in picks[:2])


def _load_rows(conn: sqlite3.Connection, window: int) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (int(window),),
    ).fetchall()


def get_three_zodiac_picks(conn: sqlite3.Connection, window: int = 16) -> List[str]:
    rows = _load_rows(conn, window)
    if not rows:
        return ["马", "蛇", "龙"]
    scores: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP}
    for idx, row in enumerate(rows):
        weight = 1.0 / (1.0 + idx * 0.12)
        nums = json.loads(row["numbers_json"])
        for n in nums:
            scores[get_zodiac_by_number(int(n))] += weight
        scores[get_zodiac_by_number(int(row["special_number"]))] += weight * 1.4
    ranked = [z for z, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    top3 = ranked[:3]
    while len(top3) < 3:
        for z in ZODIAC_MAP:
            if z not in top3:
                top3.append(z)
                break
    return top3[:3]


def backtest_zodiac_full_match(conn: sqlite3.Connection, mode: str = "strict_two", lookback: int = 60) -> Tuple[float, int]:
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC"
    ).fetchall()
    if len(rows) < 10:
        return 0.0, 0
    start = max(10, len(rows) - int(lookback))
    hits = 0
    samples = 0
    miss_streak = 0
    max_miss = 0
    for i in range(start, len(rows)):
        history = rows[max(0, i - 16):i]
        if len(history) < 8:
            continue
        picks = get_three_zodiac_picks_from_rows(history)
        nums = json.loads(rows[i]["numbers_json"])
        special = int(rows[i]["special_number"])
        winning = {get_zodiac_by_number(int(n)) for n in nums}
        winning.add(get_zodiac_by_number(special))
        if mode == "exact_2":
            hit = int(any(z in winning for z in picks[:2]))
        else:
            hit = int(all(z in winning for z in picks[:2]))
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
        else:
            miss_streak = 0
    return (hits / samples if samples else 0.0), max_miss


def get_three_zodiac_picks_from_rows(rows: Sequence[sqlite3.Row]) -> List[str]:
    scores: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP}
    for idx, row in enumerate(rows):
        weight = 1.0 / (1.0 + idx * 0.12)
        nums = json.loads(row["numbers_json"])
        for n in nums:
            scores[get_zodiac_by_number(int(n))] += weight
        scores[get_zodiac_by_number(int(row["special_number"]))] += weight * 1.4
    ranked = [z for z, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    return ranked[:3]
