#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import argparse, csv, io, json, math, os, re, socket, sqlite3, time, pickle, subprocess
from urllib.error import URLError
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

_BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_params_zodiac.json"


def load_best_zodiac_params():
    if _BEST_PARAMS_PATH.exists():
        with open(_BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


BEST_PARAMS_ZODIAC_PATH = Path(__file__).resolve().parent / "best_params_zodiac.json"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"


def load_best_params():
    for path in (BEST_PARAMS_ZODIAC_PATH, BEST_PARAMS_PATH):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

# 控制台编码统一
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 可选外部模块（不存在时提供回退）
try:
    from tail_predictor import get_best_tail, backtest_tail
except Exception:
    def get_best_tail(*args, **kwargs): return []
    def backtest_tail(*args, **kwargs): return 0.0, 0, 0

def get_three_zodiac_picks_from_history_rows(rows: Sequence[sqlite3.Row], conn=None) -> List[str]:
    return _get_three_zodiac_from_history_rows(rows, conn=conn)

try:
    from lstm_predictor import predict_lstm_proba
except ImportError:
    predict_lstm_proba = None

# 安全包装 HMM 预测，避免缺少 hmmlearn 时崩溃
def safe_get_hmm_state_proba(conn):
    try:
        from hmm_features import get_hmm_state_proba
        return get_hmm_state_proba(conn)
    except Exception:
        return None

try:
    from risk_manager import RiskManager
except Exception:
    class RiskManager:
        def __init__(self, bankroll: float = 1000.0): self.bankroll = bankroll
        def get_bet_recommendation(self, *_args, **_kwargs): return {"suspended": False, "recommended_stake": 0.0}


# 全局常量
DB_PATH_DEFAULT = str(SCRIPT_DIR / "newmacau_marksix.db")
MACAU_API_URL = "https://marksix6.net/index.php?api=1"
API_TIMEOUT_DEFAULT = 20
API_RETRIES_DEFAULT = 4
API_RETRY_BACKOFF_SECONDS = 2.0
MINED_CONFIG_KEY = "mined_strategy_config_v1"
ALL_NUMBERS = list(range(1, 50))
FEATURE_WINDOW_DEFAULT = 10

STRATEGY_BASE_WINDOWS = {
    "hot_v1": 6, "momentum_v1": 7, "cold_rebound_v1": 13,
    "balanced_v1": 10, "pattern_mined_v1": 6, "ensemble_v2": 10,
}
WEIGHT_WINDOW_DEFAULT = 30
HEALTH_WINDOW_DEFAULT = 18
BACKTEST_ISSUES_DEFAULT = 120
ZERO_HIT_TRIGGER_THRESHOLD = float(os.environ.get("ZERO_HIT_TRIGGER_THRESHOLD", "0.5"))

ENSEMBLE_DIVERSITY_BONUS = 0.18
BIAS_THRESHOLD = 0.65
BIAS_ADJUSTMENT = 0.40
FORCED_BIAS_COEFFICIENT = 0.75

STRATEGY_LABELS = {
    "balanced_v1": "组合策略", "hot_v1": "热号策略", "cold_rebound_v1": "冷号回补",
    "momentum_v1": "近期动量", "ensemble_v2": "集成投票", "pattern_mined_v1": "规律挖掘",
}
STRATEGY_IDS = ["balanced_v1", "hot_v1", "cold_rebound_v1", "momentum_v1", "ensemble_v2", "pattern_mined_v1"]
SPECIAL_ANALYSIS_ORDER = ["pattern_mined_v1", "ensemble_v2", "momentum_v1", "cold_rebound_v1", "hot_v1", "balanced_v1"]
TEXIAO5_SIZE_DEFAULT = 5

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}
ZODIAC_PAIR = {
    "鼠": "牛", "牛": "鼠", "虎": "猪", "猪": "虎", "兔": "狗", "狗": "兔",
    "龙": "鸡", "鸡": "龙", "蛇": "猴", "猴": "蛇", "马": "羊", "羊": "马"
}

# =================== 来自优化器的策略函数 (适配主脚本) ===================
def _row_numbers(r):
    if isinstance(r, dict):
        return json.loads(r["numbers_json"])
    if hasattr(r, "keys") and "numbers_json" in r.keys():
        return json.loads(r["numbers_json"])
    return json.loads(r[0]) if isinstance(r[0], str) else r[0]

def _row_special(r):
    if isinstance(r, dict):
        return int(r["special_number"])
    if hasattr(r, "keys") and "special_number" in r.keys():
        return int(r["special_number"])
    return int(r[1])

def _strategy_single_weighted(rows, wsize, rec_w, safe_th):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = rows[-wsize:] if len(rows) >= wsize else rows
    for idx, r in enumerate(recent[::-1]):
        nums = _row_numbers(r)
        sp = _row_special(r)
        w = rec_w / (1.0 + idx * 0.15)
        for n in nums: scores[get_zodiac_by_number(int(n))] += w
        scores[get_zodiac_by_number(sp)] += w * 2.0
    if max(scores.values()) < safe_th:
        omission = {z: 0 for z in ZODIAC_MAP}
        for i in range(len(recent)):
            r = recent[-(i + 1)]
            nums = _row_numbers(r)
            sp = _row_special(r)
            for z in ZODIAC_MAP:
                if omission[z] == 0: omission[z] = i + 1
            for n in nums: omission[get_zodiac_by_number(int(n))] = 0
            omission[get_zodiac_by_number(sp)] = 0
        return max(omission.items(), key=lambda x: x[1])[0]
    return max(scores.items(), key=lambda x: x[1])[0]

def _strategy_single_pure_hot(rows, wsize, rec_w, safe_th):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = rows[-wsize:] if len(rows) >= wsize else rows
    for idx, r in enumerate(recent[::-1]):
        nums = _row_numbers(r)
        w = rec_w / (1.0 + idx * 0.1)
        for n in nums: scores[get_zodiac_by_number(int(n))] += w
    return max(scores.items(), key=lambda x: x[1])[0]

def _strategy_single_pure_cold(rows, wsize, rec_w, safe_th):
    omission = {z: 0 for z in ZODIAC_MAP}
    recent = rows[-wsize:] if len(rows) >= wsize else rows
    for i, r in enumerate(recent):
        nums = _row_numbers(r)
        sp = _row_special(r)
        for z in ZODIAC_MAP: omission[z] = omission.get(z, i + 1)
        for n in nums: omission[get_zodiac_by_number(int(n))] = 0
        omission[get_zodiac_by_number(sp)] = 0
    return max(omission.items(), key=lambda x: x[1])[0]

def _strategy_single_hybrid(rows, wsize, rec_w, safe_th):
    hot = _strategy_single_pure_hot(rows, wsize, rec_w, safe_th)
    cold = _strategy_single_pure_cold(rows, wsize, rec_w, safe_th)
    scores = {z: 0.0 for z in ZODIAC_MAP}
    scores[hot] += 0.6
    scores[cold] += 0.4
    return max(scores, key=scores.get)

def _strategy_single_hot_main_only(rows, wsize, rec_w, safe_th):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = rows[-wsize:] if len(rows) >= wsize else rows
    for idx, r in enumerate(recent[::-1]):
        nums = _row_numbers(r)
        w = rec_w / (1.0 + idx * 0.1)
        for n in nums:
            scores[get_zodiac_by_number(n)] += w
    return max(scores.items(), key=lambda x: x[1])[0]

def _strategy_single_last_special(rows, wsize, rec_w, safe_th):
    if rows:
        sp = _row_special(rows[-1])
        return get_zodiac_by_number(sp)
    return "马"

def _strategy_single_cold_hot_mix(rows, wsize, rec_w, safe_th):
    hot_scores = {z: 0.0 for z in ZODIAC_MAP}
    cold_scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = rows[-wsize:] if len(rows) >= wsize else rows
    for idx, r in enumerate(recent[::-1]):
        nums = _row_numbers(r)
        w = rec_w / (1.0 + idx * 0.1)
        for n in nums:
            hot_scores[get_zodiac_by_number(n)] += w
    omission = _zodiac_omission_map_from_rows(rows)
    max_om = max(omission.values()) if omission else 1
    for z in cold_scores:
        cold_scores[z] = omission.get(z, 0) / max_om if max_om > 0 else 0
    combined = {z: (hot_scores[z] + cold_scores[z]) * 0.5 for z in ZODIAC_MAP}
    return max(combined, key=combined.get)

def _strategy_two_hot_cold(rows):
    specials = [_row_special(r) for r in rows[-10:]]
    hot_cnt = Counter([get_zodiac_by_number(sp) for sp in specials])
    hot = max(hot_cnt, key=hot_cnt.get)
    omission = {z: 0 for z in ZODIAC_MAP}
    for i, r in enumerate(rows[::-1]):
        nums = _row_numbers(r)
        sp = _row_special(r)
        for z in ZODIAC_MAP: omission[z] = omission.get(z, i + 1) if omission[z] == 0 else omission[z]
        for n in nums: omission[get_zodiac_by_number(int(n))] = 0
        omission[get_zodiac_by_number(sp)] = 0
    cold = max((z for z in ZODIAC_MAP if z != hot), key=lambda z: omission[z])
    return [hot, cold]

def _strategy_two_double_hot(rows):
    specials = [_row_special(r) for r in rows[-8:]]
    hot_cnt = Counter([get_zodiac_by_number(sp) for sp in specials])
    return [z for z, _ in hot_cnt.most_common(2)]

def _strategy_two_double_cold(rows):
    omission = {z: 0 for z in ZODIAC_MAP}
    for i, r in enumerate(rows[::-1]):
        nums = _row_numbers(r)
        sp = _row_special(r)
        for z in ZODIAC_MAP: omission[z] = omission.get(z, i + 1) if omission[z] == 0 else omission[z]
        for n in nums: omission[get_zodiac_by_number(int(n))] = 0
        omission[get_zodiac_by_number(sp)] = 0
    sorted_cold = sorted(omission.items(), key=lambda x: -x[1])
    return [sorted_cold[0][0], sorted_cold[1][0]]

def _strategy_two_last2_specials(rows):
    specials = [_row_special(r) for r in rows[:2]]
    zodiacs = [get_zodiac_by_number(sp) for sp in specials if sp is not None]
    if len(zodiacs) >= 2:
        return zodiacs[:2]
    default = ["马", "蛇"]
    while len(zodiacs) < 2:
        for z in default:
            if z not in zodiacs:
                zodiacs.append(z)
                break
    return zodiacs[:2]

def _strategy_two_hot_special_cold_main(rows):
    recent_specials = [_row_special(r) for r in rows[:8]]
    special_zodiacs = [get_zodiac_by_number(sp) for sp in recent_specials]
    hot_special = Counter(special_zodiacs).most_common(1)[0][0] if special_zodiacs else "马"
    main_zodiacs = []
    for r in rows[:12]:
        for n in json.loads(r["numbers_json"]):
            main_zodiacs.append(get_zodiac_by_number(n))
    if main_zodiacs:
        cold_main = Counter(main_zodiacs).most_common()[-1][0]
    else:
        cold_main = "龙"
    picks = [hot_special, cold_main]
    if picks[0] == picks[1]:
        if len(special_zodiacs) > 1:
            picks[1] = Counter(special_zodiacs).most_common(2)[1][0]
        else:
            picks[1] = "蛇" if picks[0] != "蛇" else "牛"
    return picks[:2]

def _strategy_two_neighbor_pair(rows):
    if not rows:
        return []
    latest_sp = _row_special(rows[0])
    latest_z = get_zodiac_by_number(latest_sp)
    pair = ZODIAC_PAIR.get(latest_z, "蛇")
    if pair == latest_z:
        pair = "蛇"
    return [latest_z, pair]

def _strategy_four_boosted(rows, four_boost):
    omission = {z: 0 for z in ZODIAC_MAP}
    specials = [_row_special(r) for r in rows]
    for i, sp in enumerate(specials[::-1]):
        z = get_zodiac_by_number(sp)
        if omission[z] == 0: omission[z] = i + 1
    for z in omission: omission[z] *= four_boost
    sorted_cold = sorted(omission.items(), key=lambda x: -x[1])
    picks = [z for z, _ in sorted_cold[:3]]
    latest_z = get_zodiac_by_number(specials[-1]) if specials else None
    if latest_z and latest_z not in picks:
        picks.append(latest_z)
    else:
        for z, _ in sorted_cold[3:]:
            if z not in picks: picks.append(z); break
    return picks[:4]

def _strategy_four_momentum(rows, momentum_w):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    for idx, r in enumerate(rows[-12:][::-1]):
        nums = _row_numbers(r)
        sp = _row_special(r)
        w = momentum_w / (1.0 + idx * 0.2)
        for n in nums: scores[get_zodiac_by_number(int(n))] += w
        scores[get_zodiac_by_number(sp)] += w * 1.5
    return [z for z, _ in sorted(scores.items(), key=lambda x: -x[1])[:4]]

def _strategy_four_hybrid(rows, four_boost, momentum_w):
    b = _strategy_four_boosted(rows, four_boost)
    m = _strategy_four_momentum(rows, momentum_w)
    union = list(dict.fromkeys(b + m))
    return union[:4]

def _strategy_four_top4_freq(rows, recent_n=12):
    counter = Counter()
    for r in rows[-recent_n:]:
        nums = _row_numbers(r)
        sp = _row_special(r)
        for n in nums:
            counter[get_zodiac_by_number(n)] += 1
        counter[get_zodiac_by_number(sp)] += 1
    return [z for z, _ in counter.most_common(4)]

def _strategy_four_cold_main_only(rows, recent_n=20):
    omission = {z: 0 for z in ZODIAC_MAP}
    for i, r in enumerate(rows[-recent_n:]):
        nums = _row_numbers(r)
        for z in ZODIAC_MAP:
            omission[z] = omission.get(z, i + 1)
        for n in nums:
            omission[get_zodiac_by_number(int(n))] = 0
    sorted_cold = sorted(omission.items(), key=lambda x: -x[1])
    return [z for z, _ in sorted_cold[:4]]

def _strategy_special_cold_neighbor(rows, zodiac_pool, params, top_n=3):
    recent_specials = [_row_special(r) for r in rows[-12:][::-1]]
    if not recent_specials: return [1,2,3]
    latest = recent_specials[0]
    omission = {}
    for i,sp in enumerate(recent_specials):
        if sp not in omission: omission[sp]=i+1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z,[])))
    if not candidates: return [1,2,3]
    cold_th = int(params.get('cold_threshold',11))
    nb1 = float(params.get('neighbor_1_bonus',6.918))
    nb2 = float(params.get('neighbor_2_bonus',0.514))
    pen = float(params.get('penalty_coeff',0.76))
    lw = float(params.get('lgb_weight',0.6146))
    ob = float(params.get('four_omit_boost',2.578))
    picks = sorted([n for n in candidates if omission.get(n,20)>=cold_th],key=lambda n:omission.get(n,20),reverse=True)[:2]
    while len(picks)<2:
        remaining = [n for n in candidates if n not in picks]
        if not remaining: break
        picks.append(max(remaining,key=lambda n:omission.get(n,20)))
    if len(picks)<top_n:
        nb = [n for n in candidates if abs(n-latest)==1 and n not in picks]
        if nb: picks.append(max(nb,key=lambda n:omission.get(n,20)+nb1))
    if len(picks)<top_n:
        nb2l = [n for n in candidates if abs(n-latest)==2 and n not in picks]
        if nb2l: picks.append(max(nb2l,key=lambda n:omission.get(n,20)+nb2))
    while len(picks)<top_n:
        rest = sorted([n for n in candidates if n not in picks],key=lambda n:omission.get(n,20),reverse=True)
        if rest: picks.append(rest[0])
        else: break
    scored=[]
    for n in picks:
        score = omission.get(n,20)*lw
        if n in recent_specials[:3]: score*=pen
        if omission.get(n,20)>=cold_th: score+=ob
        scored.append((n,score))
    scored.sort(key=lambda x:-x[1])
    return [n for n,_ in scored[:top_n]]

def _strategy_special_tail_focus(rows, zodiac_pool, params, top_n=3):
    recent_specials = [_row_special(r) for r in rows[-12:][::-1]]
    if not recent_specials: return []
    latest = recent_specials[0]
    tail_counter = Counter()
    for r in rows[-12:]:
        nums = _row_numbers(r)
        sp = _row_special(r)
        for n in nums: tail_counter[n % 10] += 1
        tail_counter[sp % 10] += 3
    hot_tails = [t for t, _ in tail_counter.most_common(6)]
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    if not candidates: return []
    picks = []
    last_tail = latest % 10
    for t in [last_tail, (last_tail + 1) % 10, (last_tail - 1) % 10, *hot_tails]:
        tail_nums = [n for n in candidates if n % 10 == t and n not in picks]
        if tail_nums:
            pick = max(tail_nums, key=lambda n: sum(1 for r in rows[-20:] if n in (_row_numbers(r)) or n == _row_special(r)))
            picks.append(pick)
        if len(picks) >= top_n: break
    while len(picks) < top_n:
        rest = [n for n in candidates if n not in picks]
        if not rest: break
        picks.append(rest[0])
    return picks[:top_n]

def _strategy_special_omission_only(rows, zodiac_pool, params, top_n=3):
    recent_specials = [_row_special(r) for r in rows[-12:][::-1]]
    if not recent_specials: return []
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission: omission[sp] = i + 1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    if not candidates: return []
    return sorted(candidates, key=lambda n: omission.get(n, 30), reverse=True)[:top_n]

def _strategy_special_zone_bias(rows, zodiac_pool, params, top_n=3):
    recent_specials = [_row_special(r) for r in rows[-12:][::-1]]
    if not recent_specials: return []
    zones = {"low": 0, "mid": 0, "high": 0}
    for sp in recent_specials:
        if sp <= 19: zones["low"] += 1
        elif sp <= 39: zones["mid"] += 1
        else: zones["high"] += 1
    target_zone = min(zones, key=zones.get)
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    if target_zone == "low": candidates = [n for n in candidates if n <= 19]
    elif target_zone == "mid": candidates = [n for n in candidates if 20 <= n <= 39]
    else: candidates = [n for n in candidates if n >= 40]
    if not candidates: return []
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission: omission[sp] = i + 1
    return sorted(candidates, key=lambda n: omission.get(n, 30), reverse=True)[:top_n]

def _strategy_special_neighbor_tail(rows, zodiac_pool, params, top_n=3):
    recent_specials = [_row_special(r) for r in rows[-12:][::-1]]
    if not recent_specials: return []
    latest = recent_specials[0]
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission: omission[sp] = i + 1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    picks = []
    nb1 = [n for n in candidates if abs(n - latest) == 1 and n not in picks]
    picks.extend(sorted(nb1, key=lambda n: omission.get(n, 30), reverse=True)[:1])
    tail = [n for n in candidates if n % 10 == latest % 10 and n not in picks]
    picks.extend(sorted(tail, key=lambda n: omission.get(n, 30), reverse=True)[:1])
    while len(picks) < top_n:
        rest = sorted([n for n in candidates if n not in picks], key=lambda n: omission.get(n, 30), reverse=True)
        if rest: picks.append(rest[0])
        else: break
    return picks[:top_n]

def _strategy_special_mixed_2cold1hot(rows, zodiac_pool, params, top_n=3):
    recent_specials = [_row_special(r) for r in rows[-12:][::-1]]
    if not recent_specials: return []
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission: omission[sp] = i + 1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    cold = sorted(candidates, key=lambda n: omission.get(n, 30), reverse=True)[:2]
    hot = sorted(candidates, key=lambda n: recent_specials.count(n), reverse=True)
    hot = [n for n in hot if n not in cold][:1]
    return cold + hot

def _strategy_special_omit_break(rows, zodiac_pool, params, top_n=3):
    recent_specials = [_row_special(r) for r in rows[-12:][::-1]]
    if not recent_specials: return []
    latest = recent_specials[0]
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission: omission[sp] = i + 1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    extreme = [n for n in candidates if omission.get(n, 30) >= 20]
    if extreme: return extreme[:top_n]
    nb1 = [n for n in candidates if abs(n - latest) == 1]
    if nb1: return nb1[:top_n]
    return sorted(candidates, key=lambda n: omission.get(n, 30), reverse=True)[:top_n]

STRATEGIES_SPECIAL = {
    "cold_neighbor": _strategy_special_cold_neighbor,
    "tail_focus": _strategy_special_tail_focus,
    "omission_only": _strategy_special_omission_only,
    "zone_bias": _strategy_special_zone_bias,
    "neighbor_tail": _strategy_special_neighbor_tail,
    "mixed_2cold1hot": _strategy_special_mixed_2cold1hot,
    "omit_break": _strategy_special_omit_break,
}

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "")

_WEIGHT_PROTECTION_PRINTED: set[str] = set()
_PROTECTION_PRINT_COUNTER = 0


@dataclass
class DrawRecord:
    issue_no: str
    draw_date: str
    numbers: List[int]
    special_number: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
           CREATE TABLE IF NOT EXISTS draws (
               issue_no TEXT PRIMARY KEY,
               draw_date TEXT NOT NULL,
               numbers_json TEXT NOT NULL,
               special_number INTEGER NOT NULL,
               source TEXT,
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL
           );

           CREATE TABLE IF NOT EXISTS prediction_runs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               issue_no TEXT NOT NULL,
               strategy TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'PENDING',
               hit_count INTEGER,
               hit_rate REAL,
               hit_count_10 INTEGER,
               hit_rate_10 REAL,
               hit_count_14 INTEGER,
               hit_rate_14 REAL,
               hit_count_20 INTEGER,
               hit_rate_20 REAL,
               special_hit INTEGER,
               created_at TEXT NOT NULL,
               reviewed_at TEXT,
               UNIQUE(issue_no, strategy)
           );

           CREATE TABLE IF NOT EXISTS prediction_picks (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id INTEGER NOT NULL,
               pick_type TEXT NOT NULL DEFAULT 'MAIN',
               number INTEGER NOT NULL,
               rank INTEGER NOT NULL,
               score REAL NOT NULL,
               reason TEXT NOT NULL,
               UNIQUE(run_id, number),
               FOREIGN KEY(run_id) REFERENCES prediction_runs(id) ON DELETE CASCADE
           );

           CREATE TABLE IF NOT EXISTS prediction_pools (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id INTEGER NOT NULL,
               pool_size INTEGER NOT NULL,
               numbers_json TEXT NOT NULL,
               created_at TEXT NOT NULL,
               UNIQUE(run_id, pool_size),
               FOREIGN KEY(run_id) REFERENCES prediction_runs(id) ON DELETE CASCADE
           );

           CREATE TABLE IF NOT EXISTS model_state (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL,
               updated_at TEXT NOT NULL
           );

           CREATE TABLE IF NOT EXISTS special_picks_log (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               issue_no TEXT NOT NULL,
               picks_json TEXT NOT NULL,
               hit_count INTEGER,
               special_hit INTEGER,
               created_at TEXT NOT NULL,
               UNIQUE(issue_no)
           );

           CREATE TABLE IF NOT EXISTS strategy_performance (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               issue_no TEXT NOT NULL,
               strategy TEXT NOT NULL,
               main_hit_count INTEGER NOT NULL,
               special_hit INTEGER NOT NULL,
               created_at TEXT NOT NULL,
               UNIQUE(issue_no, strategy)
           );
           """
    )
    _ensure_migrations(conn)
    conn.commit()


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _ensure_migrations(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "prediction_picks", "pick_type"):
        conn.execute("ALTER TABLE prediction_picks ADD COLUMN pick_type TEXT NOT NULL DEFAULT 'MAIN'")
    if not _column_exists(conn, "prediction_runs", "special_hit"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN special_hit INTEGER")
    if not _column_exists(conn, "prediction_runs", "hit_count_10"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_count_10 INTEGER")
    if not _column_exists(conn, "prediction_runs", "hit_rate_10"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_rate_10 REAL")
    if not _column_exists(conn, "prediction_runs", "hit_count_14"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_count_14 INTEGER")
    if not _column_exists(conn, "prediction_runs", "hit_rate_14"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_rate_14 REAL")
    if not _column_exists(conn, "prediction_runs", "hit_count_20"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_count_20 INTEGER")
    if not _column_exists(conn, "prediction_runs", "hit_rate_20"):
        conn.execute("ALTER TABLE prediction_runs ADD COLUMN hit_rate_20 REAL")


def get_model_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM model_state WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None


def set_model_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    now = utc_now()
    conn.execute(
        """
           INSERT INTO model_state(key, value, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
           """,
        (key, value, now),
    )


def _pick(row: Dict[str, str], keys: Sequence[str]) -> str:
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return ""


def _parse_date(date_text: str) -> Optional[str]:
    text = date_text.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_numbers(value: str) -> List[int]:
    out: List[int] = []
    for token in value.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            n = int(token)
        except ValueError:
            continue
        if 1 <= n <= 49:
            out.append(n)
    return out


def parse_macau_from_marksix6_api(payload: dict) -> List[DrawRecord]:
    records: List[DrawRecord] = []
    lottery_list = payload.get("lottery_data", [])
    if not isinstance(lottery_list, list):
        return records

    macau_data = None
    for item in lottery_list:
        if isinstance(item, dict) and item.get("name") == "新澳门彩":
            macau_data = item
            break

    if not macau_data:
        return records

    history_list = macau_data.get("history", [])
    if history_list and isinstance(history_list, list):
        for line in history_list:
            match = re.match(r"(\d{6,7})\s*期[：:]\s*([\d,]+)", line)
            if not match:
                continue
            expect_raw = match.group(1)
            numbers_str = match.group(2)
            num_list = _parse_numbers(numbers_str)
            if len(num_list) < 7:
                continue
            main_numbers = num_list[:6]
            special = num_list[6]

            if len(expect_raw) == 7:
                year = expect_raw[2:4]
                seq = str(int(expect_raw[4:]))
            elif len(expect_raw) == 6:
                year = expect_raw[:2]
                seq = str(int(expect_raw[2:]))
            else:
                continue
            issue_no = f"{year}/{seq.zfill(3)}"

            draw_date = _parse_date(macau_data.get("openTime", "").split()[0]) if macau_data.get("openTime") else None
            if not draw_date:
                draw_date = "2026-01-01"
            records.append(DrawRecord(
                issue_no=issue_no,
                draw_date=draw_date,
                numbers=main_numbers,
                special_number=special,
            ))
    else:
        expect_raw = str(macau_data.get("expect", ""))
        numbers_raw = macau_data.get("openCode") or macau_data.get("numbers")
        if numbers_raw:
            if isinstance(numbers_raw, str):
                num_list = _parse_numbers(numbers_raw)
            elif isinstance(numbers_raw, list):
                num_list = [int(x) for x in numbers_raw if str(x).isdigit()]
            else:
                num_list = []
            if len(num_list) >= 7:
                main_numbers = num_list[:6]
                special = num_list[6]
                if len(expect_raw) == 7:
                    year = expect_raw[2:4]
                    seq = str(int(expect_raw[4:]))
                elif len(expect_raw) == 6:
                    year = expect_raw[:2]
                    seq = str(int(expect_raw[2:]))
                else:
                    year = "26"
                    seq = "001"
                issue_no = f"{year}/{seq.zfill(3)}"
                draw_date = _parse_date(macau_data.get("openTime", "").split()[0]) if macau_data.get("openTime") else None
                if draw_date:
                    records.append(DrawRecord(
                        issue_no=issue_no,
                        draw_date=draw_date,
                        numbers=main_numbers,
                        special_number=special,
                    ))

    dedup: Dict[str, DrawRecord] = {}
    for r in records:
        dedup[r.issue_no] = r
    return sorted(dedup.values(), key=lambda r: (r.draw_date, r.issue_no))

# ========== 新增：从 weekendhk.com 抓取最新开奖数据 ==========
def fetch_recent_from_html() -> List[DrawRecord]:
    """尝试从 weekendhk.com 抓取最新开奖记录；失败时返回空列表。"""
    try:
        from urllib.parse import quote
        url = "https://www.weekendhk.com/" + quote("六合彩結果/")
        print(f"[抓取] 正在从 {url} 获取数据...")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        records: List[DrawRecord] = []

        # 1) 优先解析 drawData
        draw_data = None
        matches = re.findall(r"var\s+drawData\s*=\s*(\[.*?\]);", html, re.DOTALL)
        if matches:
            try:
                draw_data = json.loads(matches[0])
            except Exception:
                draw_data = None

        # 2) 兼容 next.js / 其他内嵌数据
        if draw_data is None:
            next_matches = re.findall(r'"drawData"\s*:\s*(\[.*?\])', html, re.DOTALL)
            if next_matches:
                try:
                    draw_data = json.loads(next_matches[0])
                except Exception:
                    draw_data = None

        if draw_data is None:
            print("[抓取] 未找到可解析的数据块，HTML 抓取失败")
            return []

        for item in draw_data:
            if not isinstance(item, dict):
                continue
            issue_raw = str(item.get("expect", "")).strip()
            if not issue_raw:
                continue
            issue_raw = re.sub(r"期$", "", issue_raw)
            if len(issue_raw) >= 7:
                year = issue_raw[2:4]
                seq = str(int(issue_raw[4:]))
                issue_no = f"{year}/{seq.zfill(3)}"
            elif len(issue_raw) == 6:
                year = issue_raw[:2]
                seq = str(int(issue_raw[2:]))
                issue_no = f"{year}/{seq.zfill(3)}"
            else:
                continue

            open_code = str(item.get("openCode", ""))
            numbers_raw = re.findall(r"\d+", open_code)
            if len(numbers_raw) < 7:
                continue

            main_numbers = [int(n) for n in numbers_raw[:6]]
            special_number = int(numbers_raw[6])
            draw_date = str(item.get("drawDate", "")).strip()
            draw_date = _parse_date(draw_date) if draw_date else None
            if not draw_date:
                draw_date = datetime.now().strftime("%Y-%m-%d")

            records.append(DrawRecord(
                issue_no=issue_no,
                draw_date=draw_date,
                numbers=main_numbers,
                special_number=special_number,
            ))

        print(f"[抓取] 成功获取 {len(records)} 期数据")
        return records
    except Exception as e:
        print(f"[抓取] 失败: {e}")
        return []


def fetch_records_with_fallback(limit: int = 100, timeout: int = API_TIMEOUT_DEFAULT, retries: int = API_RETRIES_DEFAULT) -> List[DrawRecord]:
    records = fetch_recent_from_html()
    if records:
        return records
    print("[同步] HTML抓取无数据，切换到API...")
    return fetch_macau_recent_records(limit=limit, timeout=timeout, retries=retries)


def fetch_macau_records(
    timeout: int = API_TIMEOUT_DEFAULT,
    retries: int = API_RETRIES_DEFAULT,
    backoff_seconds: float = API_RETRY_BACKOFF_SECONDS,
) -> List[DrawRecord]:
    req = Request(
        MACAU_API_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; macau-local/1.0)",
            "Accept": "application/json",
        },
    )

    attempts = max(1, int(retries))
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(req, timeout=int(timeout)) as resp:
                raw = resp.read().decode("utf-8-sig")
                payload = json.loads(raw)
                records = parse_macau_from_marksix6_api(payload)
                if not records:
                    raise RuntimeError("澳门彩数据解析失败，请检查API返回格式")
                return records
        except (TimeoutError, socket.timeout, URLError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[sync] API attempt {attempt}/{attempts} failed: {exc}. retry in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"澳门API请求失败，已重试 {attempts} 次。"
        f"请稍后重试，或检查网络/目标站点可用性。last_error={last_error}"
    )


def fetch_macau_recent_records(
    limit: int = 100,
    timeout: int = API_TIMEOUT_DEFAULT,
    retries: int = API_RETRIES_DEFAULT,
    backoff_seconds: float = API_RETRY_BACKOFF_SECONDS,
) -> List[DrawRecord]:
    records = fetch_macau_records(timeout=timeout, retries=retries, backoff_seconds=backoff_seconds)
    if limit > 0:
        records = records[-int(limit):]
    return records


def upsert_draw(conn: sqlite3.Connection, record: DrawRecord, source: str) -> str:
    now = utc_now()
    existing = conn.execute("SELECT issue_no FROM draws WHERE issue_no = ?", (record.issue_no,)).fetchone()
    if existing:
        conn.execute(
            """
               UPDATE draws
               SET draw_date = ?, numbers_json = ?, special_number = ?, source = ?, updated_at = ?
               WHERE issue_no = ?
               """,
            (record.draw_date, json.dumps(record.numbers), record.special_number, source, now, record.issue_no),
        )
        return "updated"
    conn.execute(
        """
           INSERT INTO draws(issue_no, draw_date, numbers_json, special_number, source, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           """,
        (record.issue_no, record.draw_date, json.dumps(record.numbers), record.special_number, source, now, now),
    )
    return "inserted"


def sync_from_records(conn: sqlite3.Connection, records: List[DrawRecord], source: str) -> Tuple[int, int, int]:
    inserted, updated = 0, 0
    for r in records:
        result = upsert_draw(conn, r, source)
        if result == "inserted":
            inserted += 1
        else:
            updated += 1
    conn.commit()
    return len(records), inserted, updated


def has_any_draw(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM draws LIMIT 1").fetchone()
    return row is not None


def parse_issue(issue_no: str) -> Optional[Tuple[str, int, int]]:
    parts = issue_no.split("/")
    if len(parts) != 2:
        return None
    year_s, seq_s = parts
    if not (year_s.isdigit() and seq_s.isdigit()):
        return None
    return year_s, int(seq_s), len(seq_s)


def issue_sort_key(issue_no: str) -> Optional[int]:
    parsed = parse_issue(issue_no)
    if not parsed:
        return None
    year_s, seq, _ = parsed
    return int(year_s) * 1000 + seq


def build_issue(year_s: str, seq: int, width: int) -> str:
    return f"{year_s}/{str(seq).zfill(width)}"


def next_issue(issue_no: str) -> str:
    parsed = parse_issue(issue_no)
    if not parsed:
        return issue_no
    year, seq, width = parsed
    return f"{year}/{str(seq + 1).zfill(width)}"


def missing_issues_since_latest(conn: sqlite3.Connection, incoming: List[DrawRecord]) -> List[str]:
    latest_row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not latest_row:
        return []

    latest_issue = str(latest_row["issue_no"])
    latest_parsed = parse_issue(latest_issue)
    latest_key = issue_sort_key(latest_issue)
    if not latest_parsed or latest_key is None:
        return []

    incoming_set = {r.issue_no for r in incoming}
    incoming_keys = [issue_sort_key(r.issue_no) for r in incoming if issue_sort_key(r.issue_no) is not None]
    if not incoming_keys:
        return []

    max_key = max(incoming_keys)
    if max_key <= latest_key:
        return []

    year_s, seq, width = latest_parsed
    missing: List[str] = []
    probe_key = latest_key
    probe_year = int(year_s)
    probe_seq = seq

    while probe_key < max_key:
        probe_seq += 1
        if probe_seq > 366:
            probe_year += 1
            probe_seq = 1
            width = 3
        issue = build_issue(str(probe_year).zfill(len(year_s)), probe_seq, width)
        probe_key = probe_year * 1000 + probe_seq
        if issue not in incoming_set:
            exists = conn.execute("SELECT 1 FROM draws WHERE issue_no = ? LIMIT 1", (issue,)).fetchone()
            if not exists:
                missing.append(issue)

    return missing


def load_recent_draws(conn: sqlite3.Connection, limit: int = 3) -> List[List[int]]:
    rows = conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [json.loads(r["numbers_json"]) for r in rows]


def _normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    values = list(score_map.values())
    mn, mx = min(values), max(values)
    if mx == mn:
        return {k: 0.0 for k in score_map}
    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def _freq_map(draws: List[List[int]]) -> Dict[int, float]:
    freq = {n: 0.0 for n in ALL_NUMBERS}
    for draw in draws:
        for n in draw:
            freq[n] += 1.0
    return freq


def _omission_map(draws: List[List[int]]) -> Dict[int, float]:
    omission = {n: float(len(draws) + 1) for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        for n in draw:
            omission[n] = min(omission[n], float(i + 1))
    return omission


def _momentum_map(draws: List[List[int]]) -> Dict[int, float]:
    m = {n: 0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        w = 1.0 / (1.0 + i)
        for n in draw:
            m[n] += w
    return m


def get_zodiac_momentum(recent_zodiacs: Sequence[str], window: int = 10) -> Dict[str, float]:
    scores = {z: 0.0 for z in ZODIAC_MAP}
    if not recent_zodiacs:
        return scores
    tail = list(recent_zodiacs)[-window:]
    for i, z in enumerate(reversed(tail)):
        if z in scores:
            scores[z] += 1.0 / (1.0 + i)
    return scores


def get_zodiac_cycle_position(recent_zodiacs: Sequence[str], zodiac: str) -> float:
    if not recent_zodiacs:
        return 999.0
    last_idx = None
    for i, z in enumerate(reversed(recent_zodiacs)):
        if z == zodiac:
            last_idx = i
            break
    if last_idx is None:
        return float(len(recent_zodiacs) + 1)
    avg_cycle = max(1.0, len(recent_zodiacs) / max(1, len(ZODIAC_MAP)))
    return float(last_idx + 1) / avg_cycle


def _pair_affinity_map(draws: List[List[int]], window: int = 3) -> Dict[int, float]:
    pair_count: Dict[Tuple[int, int], int] = {}
    for draw in draws[:window]:
        s = sorted(draw)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                key = (s[i], s[j])
                pair_count[key] = pair_count.get(key, 0) + 1

    social = {n: 0.0 for n in ALL_NUMBERS}
    for (a, b), c in pair_count.items():
        social[a] += float(c)
        social[b] += float(c)
    return social


def _zone_heat_map(draws: List[List[int]], window: int = 3) -> Dict[int, float]:
    zone_counts = [0.0] * 5
    w = draws[:window]
    if not w:
        return {n: 0.0 for n in ALL_NUMBERS}
    for draw in w:
        for n in draw:
            zone = min(4, (n - 1) // 10)
            zone_counts[zone] += 1.0
    expected = 6.0 * len(w) / 5.0
    zone_score = [expected - c for c in zone_counts]
    return {n: zone_score[min(4, (n - 1) // 10)] for n in ALL_NUMBERS}


def _adjacency_compensation_map(draws: List[List[int]], window: int = 5) -> Dict[int, float]:
    adjacency = {n: 0.0 for n in ALL_NUMBERS}
    w = draws[:window]
    if not w:
        return adjacency
    for idx, draw in enumerate(w):
        recency_w = 1.0 / (1.0 + idx * 0.35)
        for base in draw:
            for delta, bonus in ((1, 1.6), (2, 1.0), (3, 0.5)):
                for candidate in (base - delta, base + delta):
                    if 1 <= candidate <= 49:
                        adjacency[candidate] += bonus * recency_w
    return adjacency


def _pick_top_six(scores: Dict[int, float], reason: str) -> List[Tuple[int, int, float, str]]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picked: List[Tuple[int, float]] = []
    for n, s in ranked:
        if len(picked) == 6:
            break
        proposal = [pn for pn, _ in picked] + [n]
        odd_count = sum(1 for x in proposal if x % 2 == 1)
        if len(proposal) >= 4 and (odd_count == 0 or odd_count == len(proposal)):
            continue
        zone_counts: Dict[int, int] = {}
        for x in proposal:
            z = min(4, (x - 1) // 10)
            zone_counts[z] = zone_counts.get(z, 0) + 1
        if any(c >= 4 for c in zone_counts.values()):
            continue
        picked.append((n, s))
    while len(picked) < 6:
        for n, s in ranked:
            if n not in [pn for pn, _ in picked]:
                picked.append((n, s))
                break

    target_low, target_high = 95, 205
    top6 = [n for n, _ in picked[:6]]
    total = sum(top6)
    if not (target_low <= total <= target_high):
        for i in range(5, -1, -1):
            replaced = False
            for alt_n, alt_s in ranked:
                if alt_n in top6:
                    continue
                candidate = list(top6)
                candidate[i] = alt_n
                csum = sum(candidate)
                if target_low <= csum <= target_high:
                    picked[i] = (alt_n, alt_s)
                    top6 = candidate
                    replaced = True
                    break
            if replaced:
                break

    return [(n, idx + 1, s, f"{reason} score={s:.4f}") for idx, (n, s) in enumerate(picked)]


def _default_mined_config() -> Dict[str, float]:
    return {
        "window": 6.0,
        "w_freq": 0.30,
        "w_omit": 0.45,
        "w_mom": 0.15,
        "w_pair": 0.00,
        "w_zone": 0.10,
        "w_adj": 0.10,
        "special_bonus": 0.10,
    }


def _candidate_mined_configs() -> List[Dict[str, float]]:
    windows = [6, 9, 12, 18]
    weight_triplets = [
        (0.50, 0.30, 0.20),
        (0.45, 0.35, 0.20),
        (0.40, 0.40, 0.20),
        (0.35, 0.45, 0.20),
        (0.30, 0.50, 0.20),
        (0.60, 0.20, 0.20),
        (0.20, 0.60, 0.20),
        (0.40, 0.30, 0.30),
        (0.30, 0.40, 0.30),
    ]
    pair_zone_sets = [
        (0.00, 0.00),
        (0.05, 0.05),
        (0.10, 0.00),
        (0.00, 0.10),
    ]
    out: List[Dict[str, float]] = []
    for w in windows:
        for wf, wo, wm in weight_triplets:
            for wp, wz in pair_zone_sets:
                out.append(
                    {
                        "window": float(w),
                        "w_freq": wf,
                        "w_omit": wo,
                        "w_mom": wm,
                        "w_pair": wp,
                        "w_zone": wz,
                        "w_adj": 0.10,
                        "special_bonus": 0.10,
                    }
                )
    return out


def _apply_weight_config(
    draws: List[List[int]],
    config: Dict[str, float],
    reason: str,
) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:
    window_size = int(config.get("window", FEATURE_WINDOW_DEFAULT))
    window = draws[: max(3, window_size)]
    freq = _normalize(_freq_map(window))
    omission = _normalize(_omission_map(window))
    momentum = _normalize(_momentum_map(window))
    pair = _normalize(_pair_affinity_map(window, window=min(3, len(window))))
    zone = _normalize(_zone_heat_map(window, window=min(3, len(window))))
    adjacency = _normalize(_adjacency_compensation_map(window, window=min(5, len(window))))

    w_freq = float(config.get("w_freq", 0.40))
    w_omit = float(config.get("w_omit", 0.28))
    w_mom = float(config.get("w_mom", 0.16))
    w_pair = float(config.get("w_pair", 0.00))
    w_zone = float(config.get("w_zone", 0.06))
    w_adj = float(config.get("w_adj", 0.10))

    scores: Dict[int, float] = {}
    for n in ALL_NUMBERS:
        scores[n] = (
            freq[n] * w_freq
            + omission[n] * w_omit
            + momentum[n] * w_mom
            + pair[n] * w_pair
            + zone[n] * w_zone
            + adjacency[n] * w_adj
        )

    main_picks = _pick_top_six(scores, reason)
    main_set = {n for n, _, _, _ in main_picks}
    special_candidates = [(n, s) for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True) if n not in main_set]
    if not special_candidates:
        special_candidates = [(n, s) for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    special_number, special_score = special_candidates[0]
    return main_picks, special_number, special_score, scores


def mine_pattern_config_from_rows(rows: Sequence[sqlite3.Row]) -> Dict[str, float]:
    if len(rows) < 3:
        return _default_mined_config()

    candidates = _candidate_mined_configs()
    best_cfg = _default_mined_config()
    best_score = -1.0

    min_history = 3
    eval_span = min(500, len(rows) - min_history)
    start = max(min_history, len(rows) - eval_span)

    parsed_main = [json.loads(r["numbers_json"]) for r in rows]
    parsed_special = [int(r["special_number"]) for r in rows]

    for cfg in candidates:
        score_sum = 0.0
        count = 0
        for i in range(start, len(rows)):
            hist_start = max(0, i - int(cfg["window"]))
            history_desc = [parsed_main[j] for j in range(i - 1, hist_start - 1, -1)]
            if len(history_desc) < min_history:
                continue
            picks, special, _, _ = _apply_weight_config(history_desc, cfg, "规律挖掘")
            picked_main = [n for n, _, _, _ in picks]
            win_main = set(parsed_main[i])
            hit_count = len([n for n in picked_main if n in win_main])
            special_hit = 1 if int(special) == parsed_special[i] else 0
            score_sum += hit_count / 6.0 + float(cfg.get("special_bonus", 0.10)) * special_hit
            count += 1

        if count == 0:
            continue
        score = score_sum / count
        if score > best_score:
            best_score = score
            best_cfg = cfg

    return best_cfg


def ensure_mined_pattern_config(conn: sqlite3.Connection, force: bool = False) -> Dict[str, float]:
    if not force:
        cached = get_model_state(conn, MINED_CONFIG_KEY)
        if cached:
            try:
                obj = json.loads(cached)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    rows = _draws_ordered_asc(conn)
    cfg = mine_pattern_config_from_rows(rows)
    set_model_state(conn, MINED_CONFIG_KEY, json.dumps(cfg, ensure_ascii=False))
    conn.commit()
    return cfg


def _rank_vote_score(score_maps: Sequence[Dict[int, float]]) -> Dict[int, float]:
    votes = {n: 0.0 for n in ALL_NUMBERS}
    for m in score_maps:
        ranked = sorted(m.items(), key=lambda x: x[1], reverse=True)
        for rank, (n, _) in enumerate(ranked):
            votes[n] += float(49 - rank)
    return _normalize(votes)


def _build_candidate_pools(scores: Dict[int, float], main6: List[int]) -> Dict[int, List[int]]:
    ranked = [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    main_unique = []
    for n in main6:
        if n not in main_unique:
            main_unique.append(n)

    rest = [n for n in ranked if n not in main_unique]
    pool10 = main_unique + rest[: max(0, 10 - len(main_unique))]
    pool14 = main_unique + rest[: max(0, 14 - len(main_unique))]
    pool20 = main_unique + rest[: max(0, 20 - len(main_unique))]
    return {6: main_unique[:6], 10: pool10[:10], 14: pool14[:14], 20: pool20[:20]}


def _pool_hit_count(pool_numbers: Sequence[int], winning: set[int]) -> int:
    return len([n for n in pool_numbers if n in winning])


def _save_prediction_pools(conn: sqlite3.Connection, run_id: int, pools: Dict[int, List[int]]) -> None:
    conn.execute("DELETE FROM prediction_pools WHERE run_id = ?", (run_id,))
    now = utc_now()
    for pool_size, numbers in pools.items():
        conn.execute(
            """
               INSERT INTO prediction_pools(run_id, pool_size, numbers_json, created_at)
               VALUES (?, ?, ?, ?)
               """,
            (run_id, int(pool_size), json.dumps(numbers), now),
        )


def get_pool_numbers_for_run(conn: sqlite3.Connection, run_id: int, pool_size: int = 6) -> List[int]:
    row = conn.execute(
        "SELECT numbers_json FROM prediction_pools WHERE run_id = ? AND pool_size = ?",
        (run_id, int(pool_size)),
    ).fetchone()
    if not row:
        return []
    try:
        nums = json.loads(row["numbers_json"])
    except Exception:
        return []
    valid_numbers: List[int] = []
    for n in nums:
        if isinstance(n, int) and 1 <= n <= 49:
            valid_numbers.append(n)
            continue
        if isinstance(n, str) and n.isdigit():
            parsed = int(n)
            if 1 <= parsed <= 49:
                valid_numbers.append(parsed)
    return valid_numbers


def get_adaptive_strategy_window(strategy: str, conn: sqlite3.Connection) -> int:
    base = STRATEGY_BASE_WINDOWS.get(strategy, FEATURE_WINDOW_DEFAULT)
    health = get_strategy_health(conn, window=20)
    h = health.get(strategy, {})
    recent_avg = float(h.get("recent_avg_hit", 0.65))
    cold_streak = int(h.get("cold_streak", 0))

    if strategy == "cold_rebound_v1":
        rows = conn.execute(
            "SELECT numbers_json FROM draws ORDER BY draw_date DESC LIMIT 60"
        ).fetchall()
        all_nums = []
        for r in rows:
            all_nums.extend(json.loads(r["numbers_json"]))
        freq = Counter(all_nums)
        cold_count = sum(1 for n in ALL_NUMBERS if freq.get(n, 0) == 0)
        if cold_count >= 5:
            return min(20, base + 8)

    if recent_avg >= 0.95:
        return max(5, base - 2)
    elif recent_avg >= 0.80:
        return max(6, base - 1)
    elif recent_avg <= 0.55 or cold_streak >= 4:
        return min(15, base + 3)
    elif recent_avg <= 0.65:
        return min(13, base + 2)
    return base


# ========== 偏态检测函数（强制偏态模式） ==========
def detect_bias(conn: sqlite3.Connection, window: int = 10) -> Tuple[float, Dict[str, float]]:
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (window,),
    ).fetchall()
    if len(rows) < 3:
        return 0.0, {
            "forced": False,
            "zone_bias": 0.0,
            "parity_bias": 0.0,
            "hot_cold_bias": 0.0,
            "zone_dist": [0] * 5,
            "odd_ratio": 0.0,
        }

    zone_dist = [0] * 5
    odd_count = 0
    total_count = 0
    num_counter: Counter[int] = Counter()
    for row in rows:
        numbers = json.loads(row["numbers_json"])
        special = int(row["special_number"])
        for n in numbers:
            total_count += 1
            if int(n) % 2 == 1:
                odd_count += 1
            zone_dist[min(4, (int(n) - 1) // 10)] += 1
            num_counter[int(n)] += 1
        total_count += 1
        if special % 2 == 1:
            odd_count += 1
        zone_dist[min(4, (special - 1) // 10)] += 1
        num_counter[special] += 1

    total_events = sum(zone_dist) or 1
    expected = total_events / 5.0
    zone_bias = sum(abs(c - expected) for c in zone_dist) / total_events
    parity_bias = abs((odd_count / max(total_count, 1)) - 0.5) * 2.0
    hot_cold_bias = (max(num_counter.values()) - min(num_counter.values())) / max(total_events, 1)
    bias_score = min(1.0, (zone_bias + parity_bias + hot_cold_bias) / 3.0)
    return bias_score, {
        "forced": False,
        "zone_bias": round(zone_bias, 4),
        "parity_bias": round(parity_bias, 4),
        "hot_cold_bias": round(hot_cold_bias, 4),
        "zone_dist": zone_dist,
        "odd_ratio": round(odd_count / max(total_count, 1), 4),
    }


def adjust_weights_for_bias(weights: Dict[str, float], bias_score: float) -> Dict[str, float]:
    if bias_score < BIAS_THRESHOLD:
        return weights
    adjusted = weights.copy()
    cold_boost = 1 + BIAS_ADJUSTMENT * bias_score
    adjusted["cold_rebound_v1"] = weights.get("cold_rebound_v1", 0.15) * cold_boost
    adjusted["hot_v1"] = weights.get("hot_v1", 0.15) * (1 - BIAS_ADJUSTMENT * bias_score * 0.7)
    adjusted["momentum_v1"] = weights.get("momentum_v1", 0.15) * (1 - BIAS_ADJUSTMENT * bias_score * 0.5)
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted


# ========== 特别号 v4 增强版 ==========
def _generate_special_number_v4(
    conn: sqlite3.Connection,
    main_pool: List[int],
    issue_no: str
) -> Tuple[int, float, List[int]]:
    recent_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?) OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?) ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (issue_no, issue_no, issue_no, 80),
    ).fetchall()]

    latest_sp_row = conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 1"
    ).fetchone()
    latest_sp = int(latest_sp_row["special_number"]) if latest_sp_row else None

    prev_special = recent_specials[0] if recent_specials else None
    main_set = set(main_pool)

    omission = {n: 80 for n in ALL_NUMBERS}
    for i, num in enumerate(recent_specials):
        omission[num] = min(omission.get(num, 80), i + 1)

    tail_counter = Counter([n % 10 for n in recent_specials[:40]])
    coldest_tail = min(tail_counter.keys(), key=lambda t: tail_counter[t]) if tail_counter else 0

    scores = {}
    for n in ALL_NUMBERS:
        if n == latest_sp or n in main_set:
            continue
        score = 0.0
        if prev_special is not None:
            diff = abs(n - prev_special)
            if diff == 1:
                score += 8.8
            elif diff == 2:
                score += 6.6
            elif diff == 3:
                score += 3.8
        if recent_specials and n == recent_specials[0]:
            score *= 0.75
        if recent_specials and n in recent_specials[:3]:
            score *= 0.80
        omit = omission.get(n, 80)
        if omit >= 10:
            score += 8.0 * (omit / 15.0)
        elif omit >= 6:
            score += 4.5
        if n % 10 == coldest_tail:
            score += 5.0
        scores[n] = max(0.0, score)

    if not scores:
        return 1, 0.0, [2, 3, 4]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best = ranked[0][0]
    confidence = min(1.0, ranked[0][1] / 29.0)
    defenses = [n for n, _ in ranked[1:] if n not in main_set][:3]
    return best, round(confidence, 3), defenses


def _ensemble_strategy_v3_1(draws, mined_config, strategy_weights, conn, issue_no):
    sub_scores = {}
    for sub in ["hot_v1", "cold_rebound_v1", "momentum_v1", "balanced_v1", "pattern_mined_v1"]:
        _, _, _, score_map = generate_strategy(draws, sub, conn=conn, issue_no=issue_no)
        sub_scores[sub] = score_map
    voted = {n: 0.0 for n in ALL_NUMBERS}
    for score_map in sub_scores.values():
        for n, v in score_map.items():
            voted[n] += float(v)
    voted = _normalize(voted)
    main_picked = _pick_top_six(voted, "集成投票v3.1")
    main_set = {n for n, _, _, _ in main_picked}
    special_number, confidence, _ = _generate_special_number_v4(conn, main_set, issue_no)
    return main_picked, special_number, confidence, voted


def generate_strategy(
    draws: List[List[int]],
    strategy: str,
    mined_config: Optional[Dict[str, float]] = None,
    strategy_weights: Optional[Dict[str, float]] = None,
    conn: Optional[sqlite3.Connection] = None,
    issue_no: Optional[str] = None,
) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:

    window_size = STRATEGY_BASE_WINDOWS.get(strategy, FEATURE_WINDOW_DEFAULT)
    strategy_draws = draws[:window_size] if len(draws) > window_size else draws

    if strategy == "hot_v1":
        return _apply_weight_config(
            strategy_draws,
            {"window": float(window_size), "w_freq": 0.74, "w_omit": 0.06, "w_mom": 0.14, "w_zone": 0.06, "w_adj": 0.10},
            "热号策略"
        )
    elif strategy == "cold_rebound_v1":
        return _apply_weight_config(
            strategy_draws,
            {"window": float(window_size), "w_freq": 0.06, "w_omit": 0.62, "w_mom": 0.22, "w_zone": 0.05, "w_adj": 0.12},
            "冷号回补"
        )
    elif strategy == "momentum_v1":
        return _apply_weight_config(
            strategy_draws,
            {"window": float(window_size), "w_freq": 0.10, "w_omit": 0.05, "w_mom": 0.75, "w_zone": 0.05, "w_adj": 0.05},
            "近期动量"
        )
    elif strategy == "balanced_v1":
        return _apply_weight_config(
            strategy_draws,
            {
                "window": float(window_size),
                "w_freq": 0.36,
                "w_omit": 0.26,
                "w_mom": 0.18,
                "w_pair": 0.05,
                "w_zone": 0.06,
                "w_adj": 0.14,
            },
            "组合策略",
        )
    elif strategy == "pattern_mined_v1":
        cfg = mined_config or _default_mined_config()
        cfg["window"] = float(window_size)
        return _apply_weight_config(strategy_draws, cfg, "规律挖掘")
    elif strategy in ("ensemble_v2", "ensemble_v3"):
        if strategy_weights is None:
            strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT) if conn else {s: 1.0/len(STRATEGY_IDS) for s in STRATEGY_IDS}
        if conn is None:
            raise ValueError("ensemble_v2/v3 requires database connection")
        if issue_no is None:
            raise ValueError("ensemble_v2/v3 requires issue_no parameter")
        return _ensemble_strategy_v3_1(strategy_draws, mined_config, strategy_weights, conn, issue_no)

    return _apply_weight_config(
        strategy_draws,
        {
            "window": float(window_size),
            "w_freq": 0.40,
            "w_omit": 0.30,
            "w_mom": 0.20,
            "w_pair": 0.05,
            "w_zone": 0.05,
        },
        "组合策略",
    )


def generate_predictions(conn: sqlite3.Connection, issue_no: Optional[str] = None) -> str:
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row:
        raise RuntimeError("No draws found. Run sync/bootstrap first.")
    target_issue = issue_no or next_issue(row["issue_no"])
    draws = load_recent_draws(conn, FEATURE_WINDOW_DEFAULT)
    if len(draws) < 3:
        raise RuntimeError("Need at least 3 draws to generate predictions.")
    mined_cfg = ensure_mined_pattern_config(conn, force=False)

    strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT)

    for strategy in STRATEGY_IDS:
        now = utc_now()
        existing = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ?",
            (target_issue, strategy),
        ).fetchone()
        if existing:
            run_id = existing["id"]
            conn.execute(
                """
                   UPDATE prediction_runs
                   SET status='PENDING', hit_count=NULL, hit_rate=NULL,
                       hit_count_10=NULL, hit_rate_10=NULL,
                       hit_count_14=NULL, hit_rate_14=NULL,
                       hit_count_20=NULL, hit_rate_20=NULL,
                       special_hit=NULL, reviewed_at=NULL, created_at=?
                   WHERE id=?
                   """,
                (now, run_id),
            )
            conn.execute("DELETE FROM prediction_picks WHERE run_id = ?", (run_id,))
        else:
            cur = conn.execute(
                """
                   INSERT INTO prediction_runs(issue_no, strategy, status, created_at)
                   VALUES (?, ?, 'PENDING', ?)
                   """,
                (target_issue, strategy, now),
            )
            run_id = cur.lastrowid

        picks, special_number, special_score, score_map = generate_strategy(
            draws, strategy, mined_config=mined_cfg, strategy_weights=strategy_weights, conn=conn, issue_no=target_issue
        )
        main_numbers = [n for n, _, _, _ in picks]
        conn.executemany(
            """
               INSERT INTO prediction_picks(run_id, pick_type, number, rank, score, reason)
               VALUES (?, ?, ?, ?, ?, ?)
               """,
            [(run_id, "MAIN", n, rank, score, reason) for n, rank, score, reason in picks]
            + [(run_id, "SPECIAL", special_number, 1, special_score, "特别号候选")],
        )
        pools = _build_candidate_pools(score_map, main_numbers)
        _save_prediction_pools(conn, int(run_id), pools)
        conn.commit()
    return target_issue


def _draws_ordered_asc(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC"
    ).fetchall()


def run_historical_backtest(
    conn: sqlite3.Connection,
    min_history: int = 3,
    rebuild: bool = False,
    progress_every: int = 20,
    max_issues: int = BACKTEST_ISSUES_DEFAULT,
) -> Tuple[int, int]:
    draws = _draws_ordered_asc(conn)
    if len(draws) <= min_history:
        return 0, 0

    if max_issues > 0 and len(draws) > max_issues + min_history:
        draws = draws[-(max_issues + min_history):]
        print(f"[backtest] 限制回测范围为最近 {max_issues} 期（实际处理 {len(draws) - min_history} 期）", flush=True)

    if rebuild:
        conn.execute(
            """
               DELETE FROM prediction_pools
               WHERE run_id IN (SELECT id FROM prediction_runs WHERE issue_no IN (SELECT issue_no FROM draws))
               """
        )
        conn.execute(
            """
               DELETE FROM prediction_runs
               WHERE issue_no IN (SELECT issue_no FROM draws)
               """
        )
        conn.execute("DELETE FROM strategy_performance WHERE issue_no IN (SELECT issue_no FROM draws)")
        conn.commit()

    issues_processed = 0
    runs_processed = 0
    total_targets = len(draws) - min_history
    started_at = time.time()

    mined_cfg_cache: Dict[int, Dict[str, float]] = {}
    print(
        f"[backtest] start: total_issues={total_targets}, strategies_per_issue={len(STRATEGY_IDS)}, rebuild={rebuild}",
        flush=True,
    )

    for i in range(min_history, len(draws)):
        target = draws[i]
        issue_no = str(target["issue_no"])
        existing = conn.execute(
            """
               SELECT COUNT(*) AS c
               FROM prediction_runs
               WHERE issue_no = ? AND status = 'REVIEWED'
               """,
            (issue_no,),
        ).fetchone()
        if existing and int(existing["c"]) >= len(STRATEGY_IDS):
            continue

        history_desc = [
            json.loads(draws[j]["numbers_json"])
            for j in range(i - 1, max(-1, i - FEATURE_WINDOW_DEFAULT - 1), -1)
        ]
        if len(history_desc) < min_history:
            continue
        winning_main = set(json.loads(target["numbers_json"]))
        winning_special = int(target["special_number"])

        for strategy in STRATEGY_IDS:
            mined_cfg = None
            if strategy == "pattern_mined_v1":
                bucket = i // 3
                if bucket not in mined_cfg_cache:
                    mined_cfg_cache[bucket] = mine_pattern_config_from_rows(draws[:i])
                mined_cfg = mined_cfg_cache[bucket]
            main_picks, special_number, special_score, score_map = generate_strategy(
                history_desc,
                strategy,
                mined_config=mined_cfg,
                conn=conn,
                issue_no=issue_no,
            )
            picked_main = [n for n, _, _, _ in main_picks]
            pools = _build_candidate_pools(score_map, picked_main)
            hit_count = len([n for n in picked_main if n in winning_main])
            hit_rate = round(hit_count / 6.0, 4)
            hit_count_10 = _pool_hit_count(pools[10], winning_main)
            hit_count_14 = _pool_hit_count(pools[14], winning_main)
            hit_count_20 = _pool_hit_count(pools[20], winning_main)
            hit_rate_10 = round(hit_count_10 / 6.0, 4)
            hit_rate_14 = round(hit_count_14 / 6.0, 4)
            hit_rate_20 = round(hit_count_20 / 6.0, 4)
            special_hit = 1 if special_number == winning_special else 0

            now = utc_now()
            row = conn.execute(
                "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ?",
                (issue_no, strategy),
            ).fetchone()
            if row:
                run_id = int(row["id"])
                conn.execute(
                    """
                       UPDATE prediction_runs
                       SET status='REVIEWED', hit_count=?, hit_rate=?,
                           hit_count_10=?, hit_rate_10=?,
                           hit_count_14=?, hit_rate_14=?,
                           hit_count_20=?, hit_rate_20=?,
                           special_hit=?, created_at=?, reviewed_at=?
                       WHERE id=?
                       """,
                    (
                        hit_count,
                        hit_rate,
                        hit_count_10,
                        hit_rate_10,
                        hit_count_14,
                        hit_rate_14,
                        hit_count_20,
                        hit_rate_20,
                        special_hit,
                        now,
                        now,
                        run_id,
                    ),
                )
                conn.execute("DELETE FROM prediction_picks WHERE run_id = ?", (run_id,))
            else:
                cur = conn.execute(
                    """
                       INSERT INTO prediction_runs(
                         issue_no, strategy, status, hit_count, hit_rate,
                         hit_count_10, hit_rate_10, hit_count_14, hit_rate_14, hit_count_20, hit_rate_20,
                         special_hit, created_at, reviewed_at
                       )
                       VALUES (?, ?, 'REVIEWED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       """,
                    (
                        issue_no,
                        strategy,
                        hit_count,
                        hit_rate,
                        hit_count_10,
                        hit_rate_10,
                        hit_count_14,
                        hit_rate_14,
                        hit_count_20,
                        hit_rate_20,
                        special_hit,
                        now,
                        now,
                    ),
                )
                run_id = int(cur.lastrowid)

            conn.executemany(
                """
                   INSERT INTO prediction_picks(run_id, pick_type, number, rank, score, reason)
                   VALUES (?, ?, ?, ?, ?, ?)
                   """,
                [(run_id, "MAIN", n, rank, score, reason) for n, rank, score, reason in main_picks]
                + [(run_id, "SPECIAL", special_number, 1, special_score, "特别号候选")],
            )
            _save_prediction_pools(conn, run_id, pools)

            conn.execute(
                """
                   INSERT OR REPLACE INTO strategy_performance(issue_no, strategy, main_hit_count, special_hit, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   """,
                (issue_no, strategy, hit_count, special_hit, now),
            )
            runs_processed += 1

        issues_processed += 1
        if (
            issues_processed == 1
            or issues_processed == total_targets
            or (progress_every > 0 and issues_processed % progress_every == 0)
        ):
            elapsed = max(time.time() - started_at, 1e-9)
            pct = (issues_processed / total_targets) * 100.0 if total_targets > 0 else 100.0
            speed = issues_processed / elapsed
            eta = ((total_targets - issues_processed) / speed) if speed > 0 else 0.0
            print(
                f"[backtest] progress: {issues_processed}/{total_targets} ({pct:.1f}%), "
                f"runs={runs_processed}, elapsed={elapsed:.0f}s, eta={eta:.0f}s",
                flush=True,
            )

    conn.commit()
    return issues_processed, runs_processed


def review_issue(conn: sqlite3.Connection, issue_no: str) -> int:
    draw = conn.execute("SELECT numbers_json, special_number FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
    if not draw:
        return 0
    winning = set(json.loads(draw["numbers_json"]))
    winning_special = int(draw["special_number"])
    runs = conn.execute(
        "SELECT id, strategy FROM prediction_runs WHERE issue_no = ? AND status = 'PENDING'",
        (issue_no,),
    ).fetchall()
    count = 0
    for run in runs:
        run_id = run["id"]
        picks = conn.execute(
            "SELECT pick_type, number FROM prediction_picks WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        main_picked = [p["number"] for p in picks if p["pick_type"] in (None, "MAIN")]
        special_picked = [p["number"] for p in picks if p["pick_type"] == "SPECIAL"]
        pool10 = get_pool_numbers_for_run(conn, int(run_id), 10) or main_picked
        pool14 = get_pool_numbers_for_run(conn, int(run_id), 14) or main_picked
        pool20 = get_pool_numbers_for_run(conn, int(run_id), 20) or main_picked
        hit_count = len([n for n in main_picked if n in winning])
        hit_rate = round(hit_count / 6.0, 4)
        hit_count_10 = _pool_hit_count(pool10, winning)
        hit_count_14 = _pool_hit_count(pool14, winning)
        hit_count_20 = _pool_hit_count(pool20, winning)
        hit_rate_10 = round(hit_count_10 / 6.0, 4)
        hit_rate_14 = round(hit_count_14 / 6.0, 4)
        hit_rate_20 = round(hit_count_20 / 6.0, 4)
        special_hit = 1 if (special_picked and special_picked[0] == winning_special) else 0
        conn.execute(
            """
               UPDATE prediction_runs
               SET status='REVIEWED', hit_count=?, hit_rate=?,
                   hit_count_10=?, hit_rate_10=?,
                   hit_count_14=?, hit_rate_14=?,
                   hit_count_20=?, hit_rate_20=?,
                   special_hit=?, reviewed_at=?
               WHERE id=?
               """,
            (
                hit_count,
                hit_rate,
                hit_count_10,
                hit_rate_10,
                hit_count_14,
                hit_rate_14,
                hit_count_20,
                hit_rate_20,
                special_hit,
                utc_now(),
                run_id,
            ),
        )
        conn.execute(
            """
               INSERT OR REPLACE INTO strategy_performance(issue_no, strategy, main_hit_count, special_hit, created_at)
               VALUES (?, ?, ?, ?, ?)
               """,
            (issue_no, run["strategy"], hit_count, special_hit, utc_now()),
        )
        count += 1
    conn.commit()
    return count


def review_latest(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row:
        return 0
    return review_issue(conn, row["issue_no"])


def _fmt_num(n: int) -> str:
    return str(n).zfill(2)


def get_latest_draw(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1"
    ).fetchone()


def get_pending_runs(conn: sqlite3.Connection, limit: int = 12) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT id, issue_no, strategy, created_at FROM prediction_runs WHERE status='PENDING' ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def get_review_stats(conn: sqlite3.Connection, window: Optional[int] = None) -> List[sqlite3.Row]:
    if window:
        recent_issues = conn.execute(
            "SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT ?", (window,)
        ).fetchall()
        issue_list = [r['issue_no'] for r in recent_issues]
        if not issue_list:
            return []
        placeholders = ','.join('?' for _ in issue_list)
        query = f"""
            SELECT
              strategy,
              COUNT(*) AS c,
              AVG(hit_count) AS avg_hit,
              AVG(hit_rate) AS avg_rate,
              AVG(hit_count_10) AS avg_hit_10,
              AVG(hit_rate_10) AS avg_rate_10,
              AVG(hit_count_14) AS avg_hit_14,
              AVG(hit_rate_14) AS avg_rate_14,
              AVG(hit_count_20) AS avg_hit_20,
              AVG(hit_rate_20) AS avg_rate_20,
              AVG(COALESCE(special_hit, 0)) AS special_rate,
              AVG(CASE WHEN hit_count >= 1 THEN 1.0 ELSE 0.0 END) AS hit1_rate,
              AVG(CASE WHEN hit_count >= 2 THEN 1.0 ELSE 0.0 END) AS hit2_rate
            FROM prediction_runs
            WHERE status='REVIEWED' AND issue_no IN ({placeholders})
            GROUP BY strategy
            ORDER BY avg_rate DESC
        """
        rows = conn.execute(query, issue_list).fetchall()
    else:
        rows = conn.execute("""
            SELECT
              strategy,
              COUNT(*) AS c,
              AVG(hit_count) AS avg_hit,
              AVG(hit_rate) AS avg_rate,
              AVG(hit_count_10) AS avg_hit_10,
              AVG(hit_rate_10) AS avg_rate_10,
              AVG(hit_count_14) AS avg_hit_14,
              AVG(hit_rate_14) AS avg_rate_14,
              AVG(hit_count_20) AS avg_hit_20,
              AVG(hit_rate_20) AS avg_rate_20,
              AVG(COALESCE(special_hit, 0)) AS special_rate,
              AVG(CASE WHEN hit_count >= 1 THEN 1.0 ELSE 0.0 END) AS hit1_rate,
              AVG(CASE WHEN hit_count >= 2 THEN 1.0 ELSE 0.0 END) AS hit2_rate
            FROM prediction_runs
            WHERE status='REVIEWED'
            GROUP BY strategy
            ORDER BY avg_rate DESC
        """).fetchall()
    out = []
    for r in rows:
        strat = str(r["strategy"])
        ordered = conn.execute(
            """
            SELECT hit_count
            FROM prediction_runs
            WHERE status='REVIEWED' AND strategy = ?
            ORDER BY reviewed_at ASC, created_at ASC, id ASC
            """,
            (strat,),
        ).fetchall()
        miss_streak = 0
        max_miss_streak = 0
        for x in ordered:
            if int(x["hit_count"] or 0) == 0:
                miss_streak += 1
                max_miss_streak = max(max_miss_streak, miss_streak)
            else:
                miss_streak = 0
        row_dict = dict(r)
        row_dict["max_miss_streak"] = max_miss_streak
        out.append(row_dict)
    return out


def get_recent_reviews(conn: sqlite3.Connection, limit: int = 20) -> List[sqlite3.Row]:
    return conn.execute(
        """
           SELECT issue_no, strategy, hit_count, hit_rate, COALESCE(special_hit, 0) AS special_hit, reviewed_at
           FROM prediction_runs
           WHERE status='REVIEWED'
           ORDER BY reviewed_at DESC
           LIMIT ?
           """,
        (limit,),
    ).fetchall()


def get_draw_issues_desc(conn: sqlite3.Connection, limit: int = 300) -> List[str]:
    rows = conn.execute(
        "SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [str(r["issue_no"]) for r in rows]


def get_reviewed_runs_for_issue(conn: sqlite3.Connection, issue_no: str) -> List[sqlite3.Row]:
    return conn.execute(
        """
           SELECT
             id, issue_no, strategy,
             hit_count, hit_rate,
             hit_count_10, hit_rate_10,
             hit_count_14, hit_rate_14,
             hit_count_20, hit_rate_20,
             COALESCE(special_hit, 0) AS special_hit
           FROM prediction_runs
           WHERE issue_no = ? AND status = 'REVIEWED'
           ORDER BY strategy ASC
           """,
        (issue_no,),
    ).fetchall()


def get_picks_for_run(conn: sqlite3.Connection, run_id: int) -> Tuple[List[int], Optional[int]]:
    picks = conn.execute(
        "SELECT pick_type, number FROM prediction_picks WHERE run_id = ? ORDER BY rank ASC",
        (run_id,),
    ).fetchall()
    mains = [p["number"] for p in picks if p["pick_type"] in (None, "MAIN")]
    specials = [p["number"] for p in picks if p["pick_type"] == "SPECIAL"]
    return mains, (specials[0] if specials else None)


def backfill_missing_special_picks(conn: sqlite3.Connection) -> int:
    draws = load_recent_draws(conn, FEATURE_WINDOW_DEFAULT)
    if len(draws) < 3:
        return 0
    mined_cfg = ensure_mined_pattern_config(conn, force=False)

    runs = conn.execute(
        """
           SELECT id, strategy, issue_no
           FROM prediction_runs
           WHERE status='PENDING'
           """
    ).fetchall()
    patched = 0
    for run in runs:
        run_id = int(run["id"])
        existing_special = conn.execute(
            "SELECT 1 FROM prediction_picks WHERE run_id = ? AND pick_type = 'SPECIAL' LIMIT 1",
            (run_id,),
        ).fetchone()
        if existing_special:
            continue

        mains = conn.execute(
            "SELECT number FROM prediction_picks WHERE run_id = ? AND (pick_type = 'MAIN' OR pick_type IS NULL)",
            (run_id,),
        ).fetchall()
        main_set = {int(r["number"]) for r in mains}
        strategy_name = str(run["strategy"])
        run_issue = str(run["issue_no"])
        cfg = mined_cfg if strategy_name == "pattern_mined_v1" else None
        _, special_number, special_score, _ = generate_strategy(
            draws,
            strategy_name,
            mined_config=cfg,
            conn=conn,
            issue_no=run_issue,
        )

        if special_number in main_set:
            for n in ALL_NUMBERS:
                if n not in main_set:
                    special_number = n
                    break

        conn.execute(
            """
               INSERT OR IGNORE INTO prediction_picks(run_id, pick_type, number, rank, score, reason)
               VALUES (?, 'SPECIAL', ?, 1, ?, '特别号补齐')
               """,
            (run_id, special_number, float(special_score)),
        )
        patched += 1

    if patched > 0:
        conn.commit()
    return patched


def print_recommendation_sheet(conn: sqlite3.Connection, limit: int = 8) -> None:
    backfill_missing_special_picks(conn)
    rows = get_pending_runs(conn, limit=limit)
    print("\n6/10/14/20 推荐单:")
    if not rows:
        print("  (空)")
        return

    for r in rows:
        mains, special = get_picks_for_run(conn, int(r["id"]))
        pool6 = [int(n) for n in mains]
        pool10 = [int(n) for n in (get_pool_numbers_for_run(conn, int(r["id"]), 10) or pool6)]
        pool14 = [int(n) for n in (get_pool_numbers_for_run(conn, int(r["id"]), 14) or pool6)]
        pool20 = [int(n) for n in (get_pool_numbers_for_run(conn, int(r["id"]), 20) or pool6)]
        strategy_name = STRATEGY_LABELS.get(r["strategy"], r["strategy"])
        special_text = _fmt_num(special) if special is not None else "--"
        p6 = " ".join(_fmt_num(n) for n in pool6)
        p10 = " ".join(_fmt_num(n) for n in pool10)
        p14 = " ".join(_fmt_num(n) for n in pool14)
        p20 = " ".join(_fmt_num(n) for n in pool20)
        print(f"  [{r['issue_no']}] {strategy_name}")
        print(f"    6号池 : {p6} | 特别号: {special_text}")
        print(f"    10号池: {p10} | 特别号: {special_text}")
        print(f"    14号池: {p14} | 特别号: {special_text}")
        print(f"    20号池: {p20} | 特别号: {special_text}")


# ========== 动态权重相关函数 ==========
def get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT):
    rows = conn.execute("""
        SELECT sp.strategy, sp.main_hit_count, dr.draw_date
        FROM strategy_performance sp
        JOIN draws dr ON dr.issue_no = sp.issue_no
        ORDER BY dr.draw_date DESC, dr.issue_no DESC
        LIMIT ?
    """, (window,)).fetchall()

    baseline = 0.6
    weights = {s: baseline for s in STRATEGY_IDS}
    protection_msgs = []

    if rows:
        # 近期更高权重，避免长期历史稀释当前状态
        for idx, r in enumerate(rows):
            strategy = str(r["strategy"])
            if strategy not in weights:
                continue
            hit = float(r["main_hit_count"] or 0.0)
            recency_w = 1.0 / (1.0 + idx * 0.12)
            weights[strategy] += (hit / 6.0) * recency_w

    health = get_strategy_health(conn, window=HEALTH_WINDOW_DEFAULT)
    for strategy, h in health.items():
        if strategy not in weights:
            continue
        hit1_rate = float(h.get("hit1_rate", 0.0))
        cold_streak = int(h.get("cold_streak", 0))
        if strategy == "pattern_mined_v1":
            if cold_streak >= 5:
                weights[strategy] *= 0.72
            elif cold_streak >= 2:
                weights[strategy] *= 0.85
            if cold_streak >= 1:
                protection_msgs.append(f"[保护] 规律挖掘连挂 {cold_streak}，权重已平滑下调")
        elif strategy == "momentum_v1":
            if hit1_rate < 0.5:
                weights[strategy] *= 0.86
            if cold_streak >= 2:
                weights[strategy] *= 0.80
        elif strategy == "cold_rebound_v1":
            if cold_streak >= 2:
                weights[strategy] *= 0.88
        else:
            if hit1_rate < 0.52:
                weights[strategy] *= 0.92
            if cold_streak >= 2:
                weights[strategy] *= 0.84

    total = sum(weights.values()) or 1.0
    for msg in protection_msgs:
        print(msg, flush=True)
    return {k: round(v / total, 4) for k, v in weights.items()}

def get_trio_weights(conn: sqlite3.Connection, window: int = WEIGHT_WINDOW_DEFAULT) -> Tuple[float, float, float]:
    rows = conn.execute("""
           SELECT strategy, AVG(main_hit_count) as avg_hit
           FROM strategy_performance
           WHERE strategy IN ('momentum_v1', 'hot_v1', 'cold_rebound_v1')
           AND issue_no IN (SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT ?)
           GROUP BY strategy
       """, (window,)).fetchall()
    stats = {r["strategy"]: r["avg_hit"] for r in rows}
    w_mom = max(float(stats.get('momentum_v1', 0.0) or 0.0), 0.6)
    w_hot = max(float(stats.get('hot_v1', 0.0) or 0.0), 0.6)
    w_cold = max(float(stats.get('cold_rebound_v1', 0.0) or 0.0), 0.6)
    total = w_mom + w_hot + w_cold
    return w_mom/total, w_hot/total, w_cold/total


def get_strategy_health(conn: sqlite3.Connection, window: int = HEALTH_WINDOW_DEFAULT) -> Dict[str, Dict[str, float]]:
    health: Dict[str, Dict[str, float]] = {}
    for strategy in STRATEGY_IDS:
        rows = conn.execute(
            """
               SELECT hit_count
               FROM prediction_runs
               WHERE strategy = ? AND status = 'REVIEWED'
               ORDER BY reviewed_at DESC
               LIMIT ?
               """,
            (strategy, window),
        ).fetchall()
        if not rows:
            health[strategy] = {
                "samples": 0.0,
                "recent_avg_hit": 0.0,
                "hit1_rate": 0.0,
                "hit2_rate": 0.0,
                "cold_streak": 0.0,
            }
            continue

        hit_counts = [int(r["hit_count"] or 0) for r in rows]
        samples = len(hit_counts)
        hit1_rate = sum(1 for x in hit_counts if x >= 1) / samples
        hit2_rate = sum(1 for x in hit_counts if x >= 2) / samples
        recent_avg_hit = sum(hit_counts) / samples

        cold_streak = 0
        for x in hit_counts:
            if x == 0:
                cold_streak += 1
            else:
                break

        health[strategy] = {
            "samples": float(samples),
            "recent_avg_hit": float(recent_avg_hit),
            "hit1_rate": float(hit1_rate),
            "hit2_rate": float(hit2_rate),
            "cold_streak": float(cold_streak),
        }
    return health


# ========== 生肖相关函数（优化版） ==========
def get_consecutive_miss_for_pair(z1: str, z2: str) -> int:
    return 0


def get_zodiac_by_number(number: int) -> str:
    for zodiac, nums in ZODIAC_MAP.items():
        if number in nums:
            return zodiac
    return "马"


def _get_previous_issue(conn: sqlite3.Connection, current_issue: str) -> Optional[str]:
    row = conn.execute(
        """
           SELECT issue_no FROM draws 
           WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?)
              OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?)
           ORDER BY draw_date DESC, issue_no DESC 
           LIMIT 1
           """,
        (current_issue, current_issue, current_issue)
    ).fetchone()
    return row["issue_no"] if row else None


def _check_two_zodiac_hit(conn: sqlite3.Connection, issue_no: str) -> bool:
    draw = conn.execute(
        "SELECT numbers_json, special_number FROM draws WHERE issue_no = ?",
        (issue_no,)
    ).fetchone()
    if not draw:
        return False

    winning_main = json.loads(draw["numbers_json"])
    winning_special = int(draw["special_number"])
    winning_zodiacs = {get_zodiac_by_number(n) for n in winning_main}
    winning_zodiacs.add(get_zodiac_by_number(winning_special))

    rows = conn.execute(
        """
           SELECT numbers_json, special_number FROM draws 
           WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?)
              OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?)
           ORDER BY draw_date DESC, issue_no DESC 
           LIMIT ?
           """,
        (issue_no, issue_no, issue_no, 16)
    ).fetchall()
    if not rows:
        return False

    zodiac_scores = _build_zodiac_scores_from_rows(rows, decay=0.08)
    ranked = sorted(zodiac_scores.items(), key=lambda x: (-x[1], x[0]))
    if len(ranked) < 3:
        return []
    picks = [ranked[0][0], ranked[1][0]]

    return any(z in winning_zodiacs for z in picks)


def _zodiac_omission_map(rows: Sequence[sqlite3.Row]) -> Dict[str, int]:
    zodiac_omission = {z: len(rows) + 1 for z in ZODIAC_MAP.keys()}
    for i, row in enumerate(rows):
        numbers = json.loads(row["numbers_json"])
        special = int(row["special_number"])
        appeared_zodiacs = set()
        for n in numbers:
            appeared_zodiacs.add(get_zodiac_by_number(n))
        appeared_zodiacs.add(get_zodiac_by_number(special))
        for z in appeared_zodiacs:
            if zodiac_omission[z] > i + 1:
                zodiac_omission[z] = i + 1
    return zodiac_omission


def _zodiac_omission_map_from_rows(rows: Sequence[sqlite3.Row]) -> Dict[str, int]:
    if not rows:
        return {z: 0 for z in ZODIAC_MAP}
    return _zodiac_omission_map(rows)


def _build_zodiac_scores_from_rows(rows: Sequence[sqlite3.Row], decay: float = 0.08) -> Dict[str, float]:
    zodiac_scores: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP.keys()}
    omission_map = _zodiac_omission_map(rows)
    for idx, row in enumerate(rows):
        recency_w = 1.0 / (1.0 + idx * decay)
        numbers = json.loads(row["numbers_json"])
        for n in numbers:
            zodiac_scores[get_zodiac_by_number(int(n))] += 1.0 * recency_w
        zodiac_scores[get_zodiac_by_number(int(row["special_number"]))] += 1.8 * recency_w
    for z in zodiac_scores:
        omit = omission_map.get(z, len(rows))
        if omit >= 8:
            zodiac_scores[z] += 4.0
        elif omit >= 3:
            zodiac_scores[z] += omit / 6.0
    return zodiac_scores


def get_two_zodiac_picks(conn: sqlite3.Connection, issue_no: str, window: int = 16) -> List[str]:
    params = load_best_zodiac_params()
    strat = params.get("two_strategy", "hot_cold")

    target_row = conn.execute(
        "SELECT draw_date FROM draws WHERE issue_no = ?",
        (issue_no,),
    ).fetchone()
    if not target_row:
        return []
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?) OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?) ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (issue_no, issue_no, issue_no, window),
    ).fetchall()
    if not rows:
        return []

    if strat == "double_hot":
        return _strategy_two_double_hot(rows)
    elif strat == "double_cold":
        return _strategy_two_double_cold(rows)
    elif strat == "last2_specials":
        return _strategy_two_last2_specials(rows)
    elif strat == "hot_special_cold_main":
        return _strategy_two_hot_special_cold_main(rows)
    elif strat == "neighbor_pair":
        return _strategy_two_neighbor_pair(rows)
    else:
        return _strategy_two_hot_cold(rows)


def get_single_zodiac_pick(conn, issue_no, window=6):
    params = load_best_zodiac_params()
    strat = params.get("single_strategy", "weighted")
    wsize = int(params.get("wsize", window))
    rec_w = float(params.get("rec_w", 0.7339))
    safe_th = float(params.get("safe_th", 1.4589))

    target_row = conn.execute(
        "SELECT draw_date FROM draws WHERE issue_no = ?",
        (issue_no,),
    ).fetchone()
    if not target_row:
        return ""
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?) OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?) ORDER BY draw_date DESC, issue_no DESC LIMIT 16",
        (issue_no, issue_no, issue_no),
    ).fetchall()
    if not rows:
        return ""

    if strat == "weighted":
        return _strategy_single_weighted(rows, wsize, rec_w, safe_th)
    elif strat == "pure_hot":
        return _strategy_single_pure_hot(rows, wsize, rec_w, safe_th)
    elif strat == "pure_cold":
        return _strategy_single_pure_cold(rows, wsize, rec_w, safe_th)
    elif strat == "hybrid":
        return _strategy_single_hybrid(rows, wsize, rec_w, safe_th)
    elif strat == "hot_main_only":
        return _strategy_single_hot_main_only(rows, wsize, rec_w, safe_th)
    elif strat == "last_special":
        return _strategy_single_last_special(rows, wsize, rec_w, safe_th)
    elif strat == "cold_hot_mix":
        return _strategy_single_cold_hot_mix(rows, wsize, rec_w, safe_th)
    return _strategy_single_weighted(rows, wsize, rec_w, safe_th)


def get_hot_cold_zodiacs(conn: sqlite3.Connection, window: int = 12, top_n: int = 3) -> Tuple[List[str], List[str]]:
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (window,)
    ).fetchall()
    if len(rows) < window:
        return [], []
    score_counter: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP.keys()}
    for idx, row in enumerate(rows):
        recency_w = 1.0 / (1.0 + idx * 0.35)
        numbers = json.loads(row["numbers_json"])
        for n in numbers:
            score_counter[get_zodiac_by_number(n)] += 1.0 * recency_w
        special = row["special_number"]
        score_counter[get_zodiac_by_number(special)] += 1.2 * recency_w
    sorted_by_freq = sorted(score_counter.items(), key=lambda x: x[1], reverse=True)
    hot = [z for z, _ in sorted_by_freq[:top_n]]
    all_zodiacs = list(ZODIAC_MAP.keys())
    cold_candidates = [(z, score_counter.get(z, 0.0)) for z in all_zodiacs]
    cold_candidates.sort(key=lambda x: x[1])
    cold = [z for z, _ in cold_candidates[:top_n]]
    return hot, cold


def _get_two_zodiac_from_history_rows(rows: Sequence[sqlite3.Row], conn=None) -> List[str]:
    if not rows:
        return []

    params = load_best_zodiac_params()
    wsize = int(params.get("single_window", params.get("wsize", 6)))
    rec_w = float(params.get("single_recency_w", params.get("rec_w", 0.7339)))

    recent = rows[: max(4, min(wsize + 2, len(rows)))]
    scores: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP}
    for idx, r in enumerate(recent):
        w = rec_w / (1.0 + idx * 0.10)
        for n in json.loads(r["numbers_json"]):
            scores[get_zodiac_by_number(int(n))] += w * 0.55
        scores[get_zodiac_by_number(int(r["special_number"]))] += w * 1.9

    omission = _zodiac_omission_map(rows)
    for z, omit in omission.items():
        if omit >= 4:
            scores[z] += min(1.5, omit / 6.0)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    picks = [ranked[0][0]]
    for z, _ in ranked[1:]:
        if z not in picks:
            picks.append(z)
            break
    return picks[:2]


def _get_three_zodiac_from_history_rows(rows: Sequence[sqlite3.Row], conn=None) -> List[str]:
    if not rows:
        return []

    params = load_best_zodiac_params()
    lstm_seq_len = int(params.get("lstm_seq_len", 30))
    three_lstm_w = float(params.get("three_lstm_weight", 0.3))
    three_hmm_w = float(params.get("three_hmm_weight", 0.2))
    hmm_weight = float(params.get("hmm_weight", 0.2))

    zodiac_scores = _build_zodiac_scores_from_rows(rows, decay=0.06)
    recent = rows[-8:]
    recent_special_zodiacs = [get_zodiac_by_number(int(r["special_number"])) for r in recent]
    recent_main_zodiacs = [get_zodiac_by_number(int(n)) for r in recent for n in json.loads(r["numbers_json"])]
    for z, cnt in Counter(recent_special_zodiacs).items():
        zodiac_scores[z] += cnt * 2.55
    for z, cnt in Counter(recent_main_zodiacs).items():
        zodiac_scores[z] += cnt * 0.95

    if conn and predict_lstm_proba and three_lstm_w > 0.01:
        lstm_probs = predict_lstm_proba(conn, seq_len=lstm_seq_len)
        if lstm_probs:
            for z in zodiac_scores:
                zodiac_scores[z] = (1 - three_lstm_w) * zodiac_scores[z] + three_lstm_w * lstm_probs.get(z, 0.0)

    if conn and hmm_weight > 0.01:
        hmm_probs = safe_get_hmm_state_proba(conn)
        if hmm_probs:
            for z in zodiac_scores:
                zodiac_scores[z] = (1 - three_hmm_w) * zodiac_scores[z] + three_hmm_w * hmm_probs.get(z, 0.0)

    omission_zodiac = _zodiac_omission_map(rows)
    for z, omit in omission_zodiac.items():
        if omit >= 5:
            zodiac_scores[z] += min(2.1, omit / 4.0)

    ranked = sorted(zodiac_scores.items(), key=lambda x: (-x[1], x[0]))
    picks = [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else ["马", "蛇"]
    for z, _ in ranked[2:]:
        if z not in picks:
            picks.append(z)
            break
    return picks[:3]


def _get_four_zodiac_from_history_rows(rows, conn=None):
    if len(rows) < 3:
        return []
    params = load_best_zodiac_params()
    strat = params.get("four_strategy", "boosted")
    if strat == "boosted":
        four_boost = float(params.get("four_boost", 1.4221))
        return _strategy_four_boosted(rows, four_boost)
    elif strat == "momentum":
        momentum_w = float(params.get("momentum_w", 1.0))
        return _strategy_four_momentum(rows, momentum_w)
    elif strat == "hybrid":
        four_boost = float(params.get("four_boost", 1.4221))
        momentum_w = float(params.get("momentum_w", 1.0))
        return _strategy_four_hybrid(rows, four_boost, momentum_w)
    elif strat == "top4_freq":
        return _strategy_four_top4_freq(rows)
    elif strat == "cold_main_only":
        return _strategy_four_cold_main_only(rows)
    four_boost = float(params.get("four_boost", 1.4221))
    return _strategy_four_boosted(rows, four_boost)


def get_texiao4_picks(conn, issue_no, status="REVIEWED", k=TEXIAO5_SIZE_DEFAULT):
    row = conn.execute("SELECT draw_date FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
    if not row:
        return []
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?) OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?) ORDER BY draw_date DESC, issue_no DESC LIMIT 12",
        (issue_no, issue_no, issue_no),
    ).fetchall()
    if not rows:
        return list(ZODIAC_MAP.keys())[:k]
    params = load_best_zodiac_params()
    zodiac_pool = _get_four_zodiac_from_history_rows(rows, conn)
    picks = get_precise_specials_for_issue(conn, issue_no, zodiac_pool, top_n=k)
    return [get_zodiac_by_number(n) for n in picks]


def get_recent_texiao5_report(conn, lookback=10):
    rows = _draws_ordered_asc(conn)
    if len(rows) < 2:
        return {"samples":0, "hit_rate":0.0, "max_miss_streak":0}
    start = max(1, len(rows)-lookback)
    hits, samples, miss, max_miss = 0,0,0,0
    for i in range(start, len(rows)):
        issue_no = rows[i]["issue_no"]
        picks5 = get_texiao4_picks(conn, issue_no, status="REVIEWED", k=TEXIAO5_SIZE_DEFAULT)
        win_special = int(rows[i]["special_number"])
        win_zodiac = get_zodiac_by_number(win_special)
        if win_zodiac in set(picks5):
            hits += 1
            miss = 0
        else:
            miss += 1
            max_miss = max(max_miss, miss)
        samples += 1
    rate = hits/samples if samples>0 else 0.0
    return {"samples":samples, "hit_rate":rate, "max_miss_streak":max_miss}


_get_five_zodiac_from_history_rows = _get_four_zodiac_from_history_rows


def _get_single_zodiac_from_history_rows(rows: Sequence[sqlite3.Row], conn=None) -> str:
    if not rows:
        return "马"

    params = load_best_zodiac_params()
    wsize = int(params.get("single_window", params.get("wsize", 6)))
    rec_w = float(params.get("single_recency_w", params.get("rec_w", 0.7339)))
    safe_th = float(params.get("single_safe_threshold", params.get("safe_th", 1.4589)))

    scores: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP}
    recent = rows[: max(3, min(wsize, len(rows)))]
    for idx, r in enumerate(recent):
        w = rec_w / (1.0 + idx * 0.12)
        nums = json.loads(r["numbers_json"])
        for n in nums:
            scores[get_zodiac_by_number(int(n))] += w * 0.75
        scores[get_zodiac_by_number(int(r["special_number"]))] += w * 2.35

    latest_special_z = get_zodiac_by_number(int(rows[0]["special_number"]))
    scores[latest_special_z] += 0.65
    omission = _zodiac_omission_map(rows)
    for z, omit in omission.items():
        if omit >= 6:
            scores[z] += min(1.8, omit / 5.0)

    max_score = max(scores.values())
    if max_score < safe_th:
        return max(omission.items(), key=lambda x: (x[1], x[0]))[0]

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[0][0]


def get_recent_single_zodiac_report(conn: sqlite3.Connection, lookback: int = 20) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < 2:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    start = max(1, len(rows) - lookback)
    hits = 0
    samples = 0
    miss_streak = 0
    max_streak = 0
    for i in range(start, len(rows)):
        hist_rows = rows[:i]
        pick = _get_single_zodiac_from_history_rows(hist_rows, conn=conn)
        win_main = json.loads(rows[i]["numbers_json"])
        win_sp = int(rows[i]["special_number"])
        win_zod = {get_zodiac_by_number(n) for n in win_main}
        win_zod.add(get_zodiac_by_number(win_sp))
        hit = 1 if pick in win_zod else 0
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_streak = max(max_streak, miss_streak)
        else:
            miss_streak = 0
    rate = hits / samples if samples else 0.0
    return {"samples": float(samples), "hit_rate": rate, "max_miss_streak": float(max_streak)}


def get_recent_two_zodiac_report(conn: sqlite3.Connection, lookback: int = 20) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < 2:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    start = max(1, len(rows) - lookback)
    hits = 0
    samples = 0
    miss_streak = 0
    max_streak = 0
    for i in range(start, len(rows)):
        hist_rows = rows[:i]
        picks = _get_two_zodiac_from_history_rows(hist_rows, conn=conn)
        win_main = json.loads(rows[i]["numbers_json"])
        win_sp = int(rows[i]["special_number"])
        win_zod = {get_zodiac_by_number(n) for n in win_main}
        win_zod.add(get_zodiac_by_number(win_sp))
        hit = 1 if any(z in win_zod for z in picks) else 0
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_streak = max(max_streak, miss_streak)
        else:
            miss_streak = 0
    rate = hits / samples if samples else 0.0
    return {"samples": float(samples), "hit_rate": rate, "max_miss_streak": float(max_streak)}


def get_recent_three_zodiac_report(
    conn: sqlite3.Connection,
    lookback: int = 20,
    history_window: int = 16,
) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < history_window + 1:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    start = max(history_window, len(rows) - lookback)
    hits = 0
    samples = 0
    miss_streak = 0
    max_miss_streak = 0
    for i in range(start, len(rows)):
        history_rows = rows[max(0, i - history_window):i]
        if len(history_rows) < history_window:
            continue
        picks = _get_three_zodiac_from_history_rows(history_rows, conn=conn)
        win_main = json.loads(rows[i]["numbers_json"])
        win_special = int(rows[i]["special_number"])
        winning_zodiacs = {get_zodiac_by_number(int(n)) for n in win_main}
        winning_zodiacs.add(get_zodiac_by_number(win_special))
        match_count = sum(1 for z in picks if z in winning_zodiacs)
        hit = 1 if match_count >= 2 else 0
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_miss_streak = max(max_miss_streak, miss_streak)
        else:
            miss_streak = 0
    if samples == 0:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    return {
        "samples": float(samples),
        "hit_rate": float(hits / samples),
        "max_miss_streak": float(max_miss_streak),
    }


def get_recent_five_zodiac_report(
    conn: sqlite3.Connection,
    lookback: int = 20,
    history_window: int = 16,
) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < history_window + 1:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    start = max(history_window, len(rows) - lookback)
    hits = 0
    samples = 0
    miss_streak = 0
    max_miss_streak = 0
    for i in range(start, len(rows)):
        history_rows = rows[max(0, i - history_window):i]
        if len(history_rows) < history_window:
            continue
        picks = _get_five_zodiac_from_history_rows(history_rows, conn)
        win_main = json.loads(rows[i]["numbers_json"])
        win_special = int(rows[i]["special_number"])
        winning_zodiacs = {get_zodiac_by_number(int(n)) for n in win_main}
        winning_zodiacs.add(get_zodiac_by_number(win_special))
        hit = 1 if any(z in winning_zodiacs for z in picks) else 0
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_miss_streak = max(max_miss_streak, miss_streak)
        else:
            miss_streak = 0
    if samples == 0:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    return {
        "samples": float(samples),
        "hit_rate": float(hits / samples),
        "max_miss_streak": float(max_miss_streak),
    }


def get_recent_four_zodiac_report(
    conn: sqlite3.Connection,
    lookback: int = 20,
    history_window: int = 16,
) -> Dict[str, float]:
    rows = _draws_ordered_asc(conn)
    if len(rows) < history_window + 1:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    start = max(history_window, len(rows) - lookback)
    hits = 0
    samples = 0
    miss_streak = 0
    max_miss_streak = 0
    for i in range(start, len(rows)):
        history_rows = rows[max(0, i - history_window):i]
        if len(history_rows) < history_window:
            continue
        picks = _get_four_zodiac_from_history_rows(history_rows, conn)
        actual_special = get_zodiac_by_number(int(rows[i]["special_number"]))
        hit = 1 if actual_special in picks else 0
        hits += hit
        samples += 1
        if hit == 0:
            miss_streak += 1
            max_miss_streak = max(max_miss_streak, miss_streak)
        else:
            miss_streak = 0
    if samples == 0:
        return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    return {
        "samples": float(samples),
        "hit_rate": float(hits / samples),
        "max_miss_streak": float(max_miss_streak),
    }


def get_dynamic_weights_v2(conn: sqlite3.Connection, window: int = 50) -> Dict[str, float]:
    rows = conn.execute("""
        SELECT strategy,
               AVG(CASE WHEN hit_count >= 1 THEN 1.0 ELSE 0.0 END) AS hit1_rate,
               AVG(CASE WHEN hit_count >= 2 THEN 1.0 ELSE 0.0 END) AS hit2_rate,
               AVG(CASE WHEN COALESCE(hit_count, 0) = 0 THEN 1.0 ELSE 0.0 END) AS miss_rate,
               AVG(COALESCE(hit_count, 0)) AS avg_hit_count
        FROM prediction_runs
        WHERE status='REVIEWED' AND issue_no IN (
            SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT ?
        )
        GROUP BY strategy
    """, (window,)).fetchall()
    stats = {r['strategy']: dict(r) for r in rows}

    scores = {}
    for s in STRATEGY_IDS:
        r = stats.get(s, {})
        hit1 = max(float(r.get('hit1_rate') or 0.0), 0.01)
        hit2 = max(float(r.get('hit2_rate') or 0.0), 0.01)
        miss = max(float(r.get('miss_rate') or 0.0), 0.01)
        avg_hit = max(float(r.get('avg_hit_count') or 0.0), 0.01)
        score = (hit1 * 0.45) + (hit2 * 0.25) + (avg_hit * 0.20) - (miss * 0.10)
        scores[s] = max(score, 0.01)

    temp = 0.15
    exp_vals = {s: math.exp(scores[s] / temp) for s in STRATEGY_IDS}
    total = sum(exp_vals.values()) or 1.0
    return {s: v / total for s, v in exp_vals.items()}


def get_dynamic_weights(conn: sqlite3.Connection, window: int = 50) -> Dict[str, float]:
    return get_dynamic_weights_v2(conn, window)


def get_precise_specials(conn, zodiac_pool, top_n=3):
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row:
        return []
    return get_precise_specials_for_issue(conn, str(row["issue_no"]), zodiac_pool, top_n=top_n)


def get_precise_specials_for_issue(conn, issue_no, zodiac_pool, top_n=3):
    if not zodiac_pool:
        return []

    params = load_best_zodiac_params()
    strat = params.get("special_strategy", "cold_neighbor")

    target = conn.execute("SELECT draw_date FROM draws WHERE issue_no = ?", (issue_no,)).fetchone()
    if not target:
        return []
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no = ?) OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no = ?) AND issue_no < ?) ORDER BY draw_date DESC, issue_no DESC LIMIT 12",
        (issue_no, issue_no, issue_no),
    ).fetchall()
    if not rows:
        return list(ZODIAC_MAP.get(zodiac_pool[0], []))[:top_n]

    handler = STRATEGIES_SPECIAL.get(strat, _strategy_special_cold_neighbor)
    return handler(rows, zodiac_pool, params, top_n)


def _get_longest_omitted_numbers(conn, limit=6):
    all_draws = conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date DESC").fetchall()
    omission = {n: 0 for n in ALL_NUMBERS}
    for n in ALL_NUMBERS:
        omit = 0
        for row in all_draws:
            if n in json.loads(row['numbers_json']): break
            omit += 1
        omission[n] = omit
    cold_zone = {n: omission[n] for n in omission if 15 <= omission[n] <= 25}
    if len(cold_zone) >= limit:
        return sorted(cold_zone, key=cold_zone.get, reverse=True)[:limit]
    return sorted(omission, key=omission.get, reverse=True)[:limit]

def _weighted_consensus_pools(conn, issue_no):
    strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT)
    number_scores = {}
    special_scores = {}
    run_rows = conn.execute(
        "SELECT id, strategy FROM prediction_runs WHERE issue_no=? AND status='PENDING'",
        (issue_no,)
    ).fetchall()
    for idx, run in enumerate(run_rows):
        run_id = int(run["id"])
        strategy = str(run["strategy"])
        w = float(strategy_weights.get(strategy, 1.0 / max(len(STRATEGY_IDS), 1)))
        recency_w = 1.0 / (1.0 + idx * 0.15)
        pool20 = get_pool_numbers_for_run(conn, run_id, 20)
        for rank_idx, n in enumerate(pool20):
            if not (1 <= int(n) <= 49):
                continue
            rank_boost = (20 - rank_idx) / 20.0
            number_scores[int(n)] = number_scores.get(int(n), 0.0) + w * recency_w * rank_boost
        main6 = get_pool_numbers_for_run(conn, run_id, 6)
        for n in main6:
            if 1 <= int(n) <= 49:
                number_scores[int(n)] = number_scores.get(int(n), 0.0) + w * recency_w * 0.45
        _, special = get_picks_for_run(conn, run_id)
        if special is not None and 1 <= int(special) <= 49:
            special_scores[int(special)] = special_scores.get(int(special), 0.0) + w * recency_w

    if not number_scores: return [], [], [], [], None

    ranked_numbers = [n for n, _ in sorted(number_scores.items(), key=lambda x: (-x[1], x[0]))]
    pool20 = ranked_numbers[:20]
    pool14 = pool20[:14]
    pool10 = pool20[:10]
    main6 = pool20[:6]

    special = None
    if special_scores:
        special = sorted(special_scores.items(), key=lambda x: (-x[1], x[0]))[0][0]
    else:
        for n in pool20:
            if n not in main6:
                special = n
                break
    return main6, pool10, pool14, pool20, special

def get_trio_from_merged_pool20_v2(conn, issue_no):
    _, _, _, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if not pool20 or len(pool20) < 3: return []
    all_pools = []
    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=? AND status='PENDING'",
            (issue_no, strategy)
        ).fetchone()
        if run:
            p20 = get_pool_numbers_for_run(conn, run["id"], 20)
            p20_filtered = [n for n in p20 if n in pool20]
            all_pools.extend(p20_filtered)
    if len(all_pools) < 3: return pool20[:3]
    app_count = Counter(all_pools)
    diff_numbers = [n for n, c in app_count.items() if 1 <= c <= 2 and n in pool20]
    if len(diff_numbers) < 6: diff_numbers = [n for n, c in app_count.items() if c <= 3 and n in pool20]
    if len(diff_numbers) < 3: diff_numbers = pool20[:15]
    draws = load_recent_draws(conn, FEATURE_WINDOW_DEFAULT)
    if len(draws) < 3: return diff_numbers[:3]
    momentum = _momentum_map(draws); freq = _freq_map(draws); omission = _omission_map(draws)
    momentum_norm = _normalize(momentum); freq_norm = _normalize(freq); omission_norm = _normalize(omission)
    w_mom, w_hot, w_cold = get_trio_weights(conn, window=WEIGHT_WINDOW_DEFAULT)
    scores = {}
    for n in diff_numbers[:15]:
        score = (w_mom * momentum_norm.get(n, 0) + w_hot * freq_norm.get(n, 0) + w_cold * omission_norm.get(n, 0))
        score += (6 - app_count.get(n, 3)) * 0.15
        scores[n] = score
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [n for n, _ in sorted_nums[:10]]
    def is_valid(tri):
        odd_cnt = sum(1 for x in tri if x % 2 == 1)
        total = sum(tri)
        return 1 <= odd_cnt <= 2 and 80 <= total <= 130
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            for k in range(j+1, len(candidates)):
                tri = (candidates[i], candidates[j], candidates[k])
                if is_valid(tri): return list(tri)
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            for k in range(j+1, len(candidates)):
                tri = (candidates[i], candidates[j], candidates[k])
                odd_cnt = sum(1 for x in tri if x % 2 == 1)
                if 1 <= odd_cnt <= 2: return list(tri)
    return candidates[:3] if len(candidates) >= 3 else pool20[:3]

def get_final_recommendation(conn):
    row = conn.execute("SELECT issue_no FROM prediction_runs WHERE status='PENDING' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row: return None
    issue_no = row["issue_no"]
    main6, pool10, pool14, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if not main6: return None
    zodiac_single = get_single_zodiac_pick(conn, issue_no, window=16)
    zodiac_two = get_two_zodiac_picks(conn, issue_no, window=16)
    special, defs, conflict = get_special_recommendation(conn, issue_no, main6, zodiac_two)
    if special is None: return None
    strategy_specials, strategy_special_zodiacs, strategy_strong_special, strategy_strong_zodiac = get_strong_special_from_strategies(conn, issue_no, main6)
    predict_trio = get_trio_from_merged_pool20_v2(conn, issue_no)
    special_zodiacs = list(dict.fromkeys(strategy_special_zodiacs))[:4]
    while len(special_zodiacs) < 4:
        for z in ZODIAC_MAP:
            if z not in special_zodiacs:
                special_zodiacs.append(z)
                if len(special_zodiacs) == 4: break
    return (issue_no, main6, special, pool10, pool14, pool20, predict_trio,
            defs, conflict, zodiac_single, zodiac_two, special_zodiacs,
            strategy_specials, strategy_special_zodiacs, strategy_strong_special, strategy_strong_zodiac)

def print_final_recommendation(conn):
    rec = get_final_recommendation(conn)
    if not rec:
        print("\n最终推荐: (暂无有效预测)")
        return
    (issue_no, main6, special, pool10, pool14, pool20, predict_trio,
     special_defenses, special_conflict, zodiac_single, zodiac_two,
     special_zodiacs, strategy_specials, strategy_special_zodiacs,
     strategy_strong_special, strategy_strong_zodiac) = rec
    p6 = " ".join(f"{n:02d}" for n in main6)
    p10 = " ".join(f"{n:02d}" for n in pool10)
    p14 = " ".join(f"{n:02d}" for n in pool14)
    p20 = " ".join(f"{n:02d}" for n in pool20)
    trio_str = " ".join(f"{n:02d}" for n in predict_trio) if predict_trio else "无"
    special_text = f"{special:02d}"
    print()
    print(f"一生肖推荐: {zodiac_single}")
    print(f"二生肖推荐: {'、'.join(zodiac_two)}")
    three_history_rows = _draws_ordered_asc(conn)
    three_picks = _get_three_zodiac_from_history_rows(three_history_rows[:min(len(three_history_rows), 16)], conn=conn)
    print(f"三生肖推荐: {'、'.join(three_picks)}")
    print(f"特别生肖推荐: {'、'.join(special_zodiacs)}")
    latest = get_latest_draw(conn)
    if latest:
        print(f"推荐期数日期: {latest['issue_no']}（{latest['draw_date']}）")
    one_rep = get_recent_single_zodiac_report(conn, lookback=10)
    two_rep = get_recent_two_zodiac_report(conn, lookback=10)
    three_rep = get_recent_three_zodiac_report(conn, lookback=10)
    four_rep = get_recent_four_zodiac_report(conn, lookback=10)
    print(f"一生肖近10期命中率: {one_rep['hit_rate']*100:.1f}% 最大连空{int(one_rep['max_miss_streak'])}")
    print(f"二生肖近10期命中率: {two_rep['hit_rate']*100:.1f}% 最大连空{int(two_rep['max_miss_streak'])}")
    print(f"三生肖近10期命中率(至少中2个): {three_rep['hit_rate']*100:.1f}% 最大连空{int(three_rep['max_miss_streak'])}")
    print(f"特别生肖近10期命中率: {four_rep['hit_rate']*100:.1f}% 最大连空{int(four_rep['max_miss_streak'])}")
    sp_report = get_recent_special_picks_report(conn, lookback=20)
    print(f"特别号精选回测（最近20期）: 命中率={sp_report['hit_rate']*100:.1f}% 最大连空={int(sp_report['max_miss_streak'])}")
    enhanced_zodiacs = list(special_zodiacs) + [get_zodiac_by_number(int(r['special_number'])) for r in conn.execute("SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 3").fetchall()]
    enhanced_zodiacs = list(dict.fromkeys(enhanced_zodiacs))
    while len(enhanced_zodiacs) < 4:
        for z in ZODIAC_MAP:
            if z not in enhanced_zodiacs:
                enhanced_zodiacs.append(z)
                if len(enhanced_zodiacs) >= 4: break
    precise = get_precise_specials_for_issue(conn, issue_no, enhanced_zodiacs, top_n=3)
    if precise:
        ps_str = " ".join(f"{n:02d}" for n in precise)
        ps_detail = ", ".join(f"{n:02d}({get_zodiac_by_number(n)})" for n in precise)
        print(f"精选特别号 (3码): {ps_str}  ({ps_detail})")
        log_special_picks(conn, issue_no, precise, special)
    if four_rep['hit_rate'] < 0.65:
        latest_sp = conn.execute("SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()["special_number"]
        trend_z = get_zodiac_by_number(int(latest_sp))
        if trend_z not in special_zodiacs:
            special_zodiacs[-1] = trend_z
            print(f"[修正] 特别生肖即时跟随: {trend_z}")
    km = KellyManager()
    km_stake = km.kelly_stake(four_rep['hit_rate'], 1.5)
    if km_stake > 0:
        print(f"特别生肖建议仓位: {km_stake:.2f} 元")
    else:
        print(f"特别生肖建议仓位: <未达正期望>, 试探仓位 {km.bankroll*0.02:.2f} 元")
    rm = RiskManager()
    z_rec = rm.get_bet_recommendation("zodiac_three_history", 0.30, 5.0, rm.bankroll)
    s_rec = rm.get_bet_recommendation("special", 0.03, 45.0, rm.bankroll)
    print(f"风控: 生肖{'暂停' if z_rec['suspended'] else '继续'} | 特别号{'暂停' if s_rec['suspended'] else '继续'}")
    print("=" * 50)

# ========== 历史回溯专用精选函数（避免数据穿越） ==========
def get_precise_specials_from_history(history_rows, zodiac_pool, top_n=3):
    if not zodiac_pool: return []
    latest_row = history_rows[0]
    latest_special = int(latest_row['special_number'])
    recent_specials = [int(r['special_number']) for r in history_rows[:12]]
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission: omission[sp] = i + 1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    if not candidates: return []
    tail_counter = Counter()
    for row in history_rows[:8]:
        for n in json.loads(row['numbers_json']): tail_counter[n % 10] += 1
    for sp in recent_specials[:8]: tail_counter[sp % 10] += 3
    hot_tails = {t for t, _ in tail_counter.most_common(6)}
    last_tail = latest_special % 10
    neighbor_tails = {last_tail, (last_tail + 1) % 10, (last_tail - 1) % 10}
    selected = []
    penalty_nums = set(recent_specials[:2])
    neighbors = [n for n in candidates if abs(n - latest_special) == 1 and n not in penalty_nums]
    if not neighbors: neighbors = [n for n in candidates if abs(n - latest_special) == 1]
    if neighbors: selected.append(max(neighbors, key=lambda n: omission.get(n, 20)))
    if len(selected) < top_n:
        tail_candidates = [n for n in candidates if n not in selected and n % 10 == last_tail and n not in penalty_nums]
        if not tail_candidates: tail_candidates = [n for n in candidates if n not in selected and n % 10 in neighbor_tails and n not in penalty_nums]
        if not tail_candidates: tail_candidates = [n for n in candidates if n not in selected and n % 10 in hot_tails and n not in penalty_nums]
        if not tail_candidates: tail_candidates = [n for n in candidates if n not in selected and n % 10 == last_tail]
        if tail_candidates: selected.append(max(tail_candidates, key=lambda n: omission.get(n, 20)))
    if len(selected) < top_n:
        neighbors2 = [n for n in candidates if abs(n - latest_special) == 2 and n not in selected and n not in penalty_nums]
        if not neighbors2: neighbors2 = [n for n in candidates if abs(n - latest_special) == 2 and n not in selected]
        if neighbors2: selected.append(max(neighbors2, key=lambda n: omission.get(n, 20)))
    if len(selected) < top_n:
        cold_pool = [n for n in candidates if n not in selected and n != latest_special]
        if cold_pool: selected.append(max(cold_pool, key=lambda n: omission.get(n, 20)))
    if len(selected) < top_n:
        remaining = [n for n in candidates if n not in selected]
        remaining.sort(key=lambda n: omission.get(n, 20), reverse=True)
        for n in remaining:
            selected.append(n)
            if len(selected) >= top_n: break
    return selected[:top_n]


def log_special_picks(conn: sqlite3.Connection, issue_no: str, picks: Sequence[int], special_number: Optional[int] = None) -> None:
    try:
        special_hit = 0
        if special_number is not None:
            special_hit = 1 if int(special_number) in {int(n) for n in picks} else 0
        conn.execute(
            "INSERT OR REPLACE INTO special_picks_log(issue_no, picks_json, hit_count, special_hit, created_at) VALUES (?, ?, ?, ?, ?)",
            (issue_no, json.dumps([int(n) for n in picks], ensure_ascii=False), special_hit, special_hit, utc_now()),
        )
        conn.commit()
    except Exception:
        pass


def get_recent_special_picks_report(conn: sqlite3.Connection, lookback: int = 20) -> Dict[str, float]:
    rows = conn.execute(
        "SELECT issue_no, picks_json, special_hit FROM special_picks_log ORDER BY id DESC LIMIT ?",
        (lookback,),
    ).fetchall()
    if not rows: return {"samples": 0.0, "hit_rate": 0.0, "max_miss_streak": 0.0}
    hits = 0
    miss_streak = 0
    max_miss = 0
    for row in rows:
        hit = int(row["special_hit"] or 0)
        hits += hit
        if hit == 0:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
        else:
            miss_streak = 0
    samples = len(rows)
    return {"samples": float(samples), "hit_rate": float(hits / samples), "max_miss_streak": float(max_miss)}

# ========== 历史回溯命令 ==========
def backfill_special_picks_log(conn, max_issues=100):
    draws = conn.execute(
        "SELECT issue_no, draw_date, special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
    if len(draws) < 16:
        print("数据不足，无法回溯（至少需要16期）。")
        return 0

    count = 0
    for i in range(12, len(draws)):
        target_issue = draws[i]['issue_no']
        target_date = draws[i]['draw_date']

        existing = conn.execute(
            "SELECT 1 FROM special_picks_log WHERE issue_no = ?", (target_issue,)
        ).fetchone()
        if existing:
            continue

        history = conn.execute(
            """SELECT numbers_json, special_number FROM draws 
               WHERE draw_date < ? OR (draw_date = ? AND issue_no < ?)
               ORDER BY draw_date DESC, issue_no DESC
               LIMIT 16""",
            (target_date, target_date, target_issue)
        ).fetchall()
        if len(history) < 12:
            continue

        base_four = _get_four_zodiac_from_history_rows(history, conn)
        recent_zodiacs = [get_zodiac_by_number(int(r['special_number'])) for r in history[:8]]
        zodiac_freq = Counter(recent_zodiacs)
        specials_hist = [int(r['special_number']) for r in history[:30]]
        omission_zodiac = {z: 0 for z in ZODIAC_MAP}
        for i, sp in enumerate(specials_hist):
            z = get_zodiac_by_number(sp)
            if omission_zodiac[z] == 0:
                omission_zodiac[z] = i + 1
        sorted_omit = sorted(omission_zodiac.items(), key=lambda x: -x[1])
        extra_freq = [z for z, _ in zodiac_freq.most_common(3) if z not in base_four][:2]
        extra_cold = [z for z, _ in sorted_omit if z not in base_four and z not in extra_freq][:2]
        last3_zodiacs = [get_zodiac_by_number(int(r['special_number'])) for r in history[:3]]
        latest_main = json.loads(history[0]['numbers_json'])
        main_counter = Counter(get_zodiac_by_number(n) for n in latest_main)
        top_main = main_counter.most_common(1)[0][0] if main_counter else None
        zodiac_pool = base_four + extra_freq + extra_cold + last3_zodiacs + ([top_main] if top_main else [])
        seen = set()
        final_pool = []
        for z in zodiac_pool:
            if z not in seen:
                seen.add(z)
                final_pool.append(z)
        while len(final_pool) < 8:
            for z in ZODIAC_MAP:
                if z not in final_pool:
                    final_pool.append(z)
                if len(final_pool) == 8: break
        zodiac_pool = final_pool[:8]

        picks = get_precise_specials_from_history(history, zodiac_pool, top_n=3)
        if picks:
            actual_special_row = conn.execute(
                "SELECT special_number FROM draws WHERE issue_no = ?", (target_issue,)
            ).fetchone()
            actual_special = actual_special_row['special_number'] if actual_special_row else None
            special_hit = 1 if actual_special is not None and actual_special in picks else 0

            conn.execute(
                "INSERT OR IGNORE INTO special_picks_log (issue_no, picks_json, special_hit, created_at) VALUES (?, ?, ?, ?)",
                (target_issue, json.dumps(picks), special_hit, utc_now())
            )
            count += 1
        if count >= max_issues:
            break

    conn.commit()
    return count


class KellyManager:
    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll
        self.loss_streak = 0

    def update_result(self, net_profit: float):
        if net_profit <= 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
        self.bankroll += net_profit
        if self.bankroll < 0:
            self.bankroll = 0.0

    def kelly_stake(self, win_rate: float, odds: float, fraction: float = 0.5) -> float:
        b = odds - 1.0
        if win_rate <= 0 or b <= 0:
            return 0.0
        f = (win_rate * b - (1 - win_rate)) / b
        f = f * fraction
        if self.loss_streak >= 2:
            f *= 0.5
        f = min(f, 0.25)
        return max(0.0, f * self.bankroll)


def get_special_recommendation(conn: sqlite3.Connection, issue_no: str, main6: Sequence[int], zodiac_two: Optional[Sequence[str]] = None) -> Tuple[Optional[int], List[int], bool]:
    top_votes = get_top_special_votes(conn, issue_no, top_n=8)
    if not top_votes:
        return None, [], False

    mains = {int(n) for n in main6}
    recent_3_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 3"
    ).fetchall()]
    recent_12_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 12"
    ).fetchall()]
    recent_8_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 8"
    ).fetchall()]

    vote_scores = Counter(top_votes)
    candidates = sorted(set(top_votes) | set(recent_12_specials) | set(recent_8_specials))
    if zodiac_two:
        allowed = set()
        for z in zodiac_two:
            allowed.update(ZODIAC_MAP.get(z, []))
        filtered = [n for n in candidates if n in allowed]
        if filtered:
            candidates = filtered
    combined = []
    for n in candidates:
        if n in mains:
            continue
        score = vote_scores.get(n, 0) * 4.0
        if recent_12_specials:
            recent_special_tail = recent_12_specials[0] % 10
            recent_special_zone = (recent_12_specials[0] - 1) // 10
            if n % 10 == recent_special_tail: score += 0.45
            if (n - 1) // 10 == recent_special_zone: score += 0.25
        if n in recent_3_specials:
            score *= 0.8949
        combined.append((n, score))

    if not combined:
        return None, [], False
    combined.sort(key=lambda x: (-x[1], x[0]))
    primary = int(combined[0][0])
    conflict = primary in mains
    defenses = []
    for n, _ in combined[1:]:
        n_int = int(n)
        if n_int == primary or n_int in defenses or n_int in mains or n_int in recent_3_specials:
            continue
        defenses.append(n_int)
        if len(defenses) >= 3:
            break
    return primary, defenses, conflict


def get_strong_special_from_strategies(
    conn: sqlite3.Connection,
    issue_no: str,
    main6: Sequence[int],
) -> Tuple[List[int], List[str], Optional[int], Optional[str]]:
    strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT)
    specials: List[int] = []
    weighted_items: List[Tuple[int, float]] = []
    for strategy in SPECIAL_ANALYSIS_ORDER:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (issue_no, strategy),
        ).fetchone()
        if not run:
            continue
        _, sp = get_picks_for_run(conn, int(run["id"]))
        if sp is None:
            continue
        special_num = int(sp)
        specials.append(special_num)
        weighted_items.append((special_num, float(strategy_weights.get(strategy, 1.0 / max(len(STRATEGY_IDS), 1)))))
    if not specials:
        return [], [], None, None

    zodiac_list = [get_zodiac_by_number(n) for n in specials]
    zodiac_counter = Counter(zodiac_list)
    number_votes = Counter(specials)
    weighted_scores: Dict[int, float] = {}
    for n, w in weighted_items:
        weighted_scores[n] = weighted_scores.get(n, 0.0) + w

    recent_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 20"
    ).fetchall()]
    omission = {n: 31 for n in ALL_NUMBERS}
    for idx, n in enumerate(recent_specials):
        omission[n] = min(omission.get(n, 31), idx + 1)

    recent_special_zodiacs = [get_zodiac_by_number(n) for n in recent_specials[:8]]
    recent_main_zodiacs: List[str] = []
    for row in conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 8"
    ).fetchall():
        recent_main_zodiacs.extend(get_zodiac_by_number(int(n)) for n in json.loads(row["numbers_json"]))
    recent_zodiac_counter = Counter(recent_special_zodiacs + recent_main_zodiacs)

    model_score: Dict[str, float] = {z: 0.0 for z in ZODIAC_MAP.keys()}
    for z, cnt in zodiac_counter.items():
        model_score[z] += cnt * 2.2
    for idx, z in enumerate(recent_special_zodiacs[:8]):
        model_score[z] += max(0.7, 2.2 - idx * 0.25)
    for idx, z in enumerate(recent_main_zodiacs[:36]):
        model_score[z] += max(0.12, 0.8 - idx * 0.02)
    hot_special = [z for z, _ in Counter(recent_special_zodiacs).most_common(1)]
    for z in hot_special:
        model_score[z] += 1.6
    omission_zodiac: Dict[str, int] = {z: 0 for z in ZODIAC_MAP.keys()}
    for idx, sp in enumerate(recent_specials):
        oz = get_zodiac_by_number(sp)
        omission_zodiac[oz] = max(omission_zodiac.get(oz, 0), 24 - idx)
    cold_zodiacs = [z for z, _ in sorted(omission_zodiac.items(), key=lambda x: (-x[1], x[0]))[:2]]
    for z in cold_zodiacs:
        model_score[z] += 3.4
    for z in ZODIAC_MAP.keys():
        if omission_zodiac.get(z, 0) >= 5:
            model_score[z] += 1.6
    ranked_zodiacs = sorted(model_score.items(), key=lambda x: (-x[1], x[0]))
    top_zodiacs = [z for z, _ in ranked_zodiacs[:4]]
    if len(top_zodiacs) < 4:
        for z, _ in ranked_zodiacs:
            if z not in top_zodiacs:
                top_zodiacs.append(z)
            if len(top_zodiacs) == 4:
                break

    mains = {int(x) for x in main6}
    candidate_scores: Dict[int, float] = {}
    for n in sorted(set(specials)):
        zodiac = get_zodiac_by_number(n)
        if zodiac not in top_zodiacs:
            continue
        score = 0.0
        score += number_votes.get(n, 0) * 2.4
        score += weighted_scores.get(n, 0.0) * 1.6
        score += zodiac_counter.get(zodiac, 0) * 1.0
        score += min(1.2, float(omission.get(n, 31)) / 24.0)
        if n in mains:
            score -= 0.8
        if zodiac in hot_special:
            score += 0.9
        if zodiac in cold_zodiacs:
            score += 0.6
        candidate_scores[n] = score

    ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
    best: Optional[int] = None
    for n, _ in ranked:
        if n not in mains:
            best = n
            break
    if best is None and ranked:
        best = ranked[0][0]
    if best is None:
        return specials, top_zodiacs, None, None
    return specials, top_zodiacs, best, get_zodiac_by_number(best)


def get_top_special_votes(conn: sqlite3.Connection, issue_no: str, top_n: int = 3) -> List[int]:
    all_specials = []
    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (issue_no, strategy)
        ).fetchone()
        if run:
            _, sp = get_picks_for_run(conn, run["id"])
            if sp is not None:
                all_specials.append(sp)
    if not all_specials:
        return []
    vote_counter = Counter(all_specials)
    sorted_items = sorted(vote_counter.items(), key=lambda x: (-x[1], x[0]))
    return [num for num, _ in sorted_items[:top_n]]


def send_pushplus_notification(title: str, content: str) -> bool:
    if not PUSHPLUS_TOKEN:
        print("[推送] 未配置 PUSHPLUS_TOKEN，跳过推送")
        return False
    import urllib.request
    import urllib.parse
    url = "https://www.pushplus.plus/send"
    data = {
        "token": PUSHPLUS_TOKEN,
        "title": title,
        "content": content,
        "template": "txt"
    }
    post_data = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=post_data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("code") == 200:
                print("[推送] 成功")
                return True
            else:
                print(f"[推送] 失败: {result}")
                return False
    except Exception as e:
        print(f"[推送] 异常: {e}")
        return False


def review_latest_prediction(conn: sqlite3.Connection) -> str:
    latest_draw = get_latest_draw(conn)
    if not latest_draw:
        return "暂无开奖数据。"
    issue_no = latest_draw["issue_no"]
    draw_date = latest_draw["draw_date"]
    actual_numbers = set(json.loads(latest_draw["numbers_json"]))
    actual_special = int(latest_draw["special_number"])
    actual_main_str = " ".join(_fmt_num(n) for n in sorted(actual_numbers))
    actual_special_str = _fmt_num(actual_special)

    runs = conn.execute(
        "SELECT id, strategy FROM prediction_runs WHERE issue_no = ? AND status='REVIEWED'",
        (issue_no,)
    ).fetchall()
    if not runs:
        return f"最新一期 {issue_no} 无预测记录（可能未运行预测）。"

    lines = []
    lines.append(f"复盘最新一期 {issue_no}（{draw_date}）")
    lines.append(f"实际开奖: 主号 {actual_main_str}  特别号 {actual_special_str}")
    lines.append("")
    lines.append("各策略预测与命中情况：")
    for run in runs:
        strategy = run["strategy"]
        strategy_name = STRATEGY_LABELS.get(strategy, strategy)
        main6, special = get_picks_for_run(conn, run["id"])
        if not main6:
            continue
        hit_count = len([n for n in main6 if n in actual_numbers])
        special_hit = 1 if special == actual_special else 0
        main_str = " ".join(_fmt_num(n) for n in main6)
        special_str = _fmt_num(special) if special is not None else "--"
        lines.append(f"  {strategy_name}: 主号 {main_str} | 特别号 {special_str} | 中主号 {hit_count}/6 | 中特别号 {'Y' if special_hit else 'N'}")
    lines.append("")
    return "\n".join(lines)


def print_dashboard(conn: sqlite3.Connection, recent_online: Optional[List[DrawRecord]] = None) -> None:
    print("\n========== 线上最新开奖 ==========")
    if recent_online:
        latest_online = recent_online[-1]
        print(
            f"最新开奖(线上): {latest_online.issue_no} {latest_online.draw_date} | "
            f"主号: {' '.join(_fmt_num(n) for n in latest_online.numbers)} | 特别号: {_fmt_num(latest_online.special_number)}"
        )
        print("最近10期(线上):")
        for r in recent_online[-10:][::-1]:
            print(
                f"  {r.issue_no} {r.draw_date} | 主号: {' '.join(_fmt_num(n) for n in r.numbers)} | 特别号: {_fmt_num(r.special_number)}"
            )
    else:
        latest = get_latest_draw(conn)
        if latest:
            nums = " ".join(_fmt_num(n) for n in json.loads(latest["numbers_json"]))
            print(f"最新开奖: {latest['issue_no']} {latest['draw_date']} | 主号: {nums} | 特别号: {_fmt_num(int(latest['special_number']))}")
        else:
            print("暂无开奖数据。")

    print("\n========== 当前推荐 ==========")
    print_recommendation_sheet(conn, limit=8)
    print_final_recommendation(conn)

    print("\n========== 复盘与健康度 ==========")
    print("策略最近10期平均命中率:")
    stats_10 = get_review_stats(conn, window=10)
    if not stats_10:
        print("  (近期暂无复盘数据，请先运行 sync)")
    for s in stats_10:
        strategy_name = STRATEGY_LABELS.get(s["strategy"], s["strategy"])
        print(
            f"  - {strategy_name}: 次数={s['c']} 平均命中={s['avg_hit']:.2f} "
            f"命中率6={s['avg_rate'] * 100:.2f}% 10={float(s['avg_rate_10'] or 0) * 100:.2f}% "
            f"14={float(s['avg_rate_14'] or 0) * 100:.2f}% 20={float(s['avg_rate_20'] or 0) * 100:.2f}% "
            f"特别号命中率={s['special_rate'] * 100:.2f}% 至少中1个={s['hit1_rate'] * 100:.2f}% 至少中2个={s['hit2_rate'] * 100:.2f}% "
            f"最大连空={int(s.get('max_miss_streak', 0))}"
        )

    confidence = 0.0
    max_miss = 0
    if stats_10:
        confidence = max(float(s.get('hit1_rate', 0.0)) for s in stats_10) * 100.0
        max_miss = max(int(s.get('max_miss_streak', 0)) for s in stats_10)
    if confidence >= 80 and max_miss < 3:
        advice = "🔥 高信心：可适当加大投入"
    elif confidence >= 60:
        advice = "👍 中等信心：正常投入"
    else:
        advice = "⚠️ 低信心：建议减少投入或观望"
    print(f"\n信心指数: {confidence:.1f}/100 | 建议投入: {advice}")

    print(f"\n策略健康度（最近{HEALTH_WINDOW_DEFAULT}期）:")
    weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT)
    health = get_strategy_health(conn, window=HEALTH_WINDOW_DEFAULT)
    for strategy in STRATEGY_IDS:
        strategy_name = STRATEGY_LABELS.get(strategy, strategy)
        h = health.get(strategy, {})
        samples = int(h.get("samples", 0.0))
        avg_hit = float(h.get("recent_avg_hit", 0.0))
        hit1 = float(h.get("hit1_rate", 0.0)) * 100.0
        hit2 = float(h.get("hit2_rate", 0.0)) * 100.0
        cold = int(h.get("cold_streak", 0.0))
        weight = float(weights.get(strategy, 0.0)) * 100.0
        print(
            f"  - {strategy_name}: 样本={samples} 最近均中={avg_hit:.2f} "
            f"近1中率={hit1:.1f}% 近2中率={hit2:.1f}% 连挂={cold} 当前权重={weight:.1f}%"
        )

    one_rep = get_recent_single_zodiac_report(conn, lookback=10)
    two_rep = get_recent_two_zodiac_report(conn, lookback=10)
    four_rep = get_recent_four_zodiac_report(conn, lookback=10, history_window=16)
    texiao5_rep = get_recent_texiao5_report(conn, lookback=10)
    sp_rep = get_recent_special_picks_report(conn, lookback=10)

    print(f"近10期: 一生肖={one_rep['hit_rate']:.3f}(连空{int(one_rep['max_miss_streak'])}) "
          f"二肖={two_rep['hit_rate']:.3f}(连空{int(two_rep['max_miss_streak'])}) "
          f"四肖={four_rep['hit_rate']:.3f}(连空{int(four_rep['max_miss_streak'])}) "
          f"特别号={sp_rep['hit_rate']:.3f}(连空{int(sp_rep.get('max_miss_streak',0))})")
    print(f"特五肖(仅特别号) 近10期命中率: {texiao5_rep['hit_rate']:.3f} 最大连空{int(texiao5_rep['max_miss_streak'])}")
    if one_rep['hit_rate'] >= 0.9 and two_rep['hit_rate'] >= 0.8 and four_rep['hit_rate'] >= 1.0:
        print("🎉 达标！")

    print("\n========== 复盘记录 ==========")
    print(review_latest_prediction(conn))

    if PUSHPLUS_TOKEN:
        rec = get_final_recommendation(conn)
        if rec:
            (issue_no, main6, special, _, _, _, predict_trio,
             special_defenses, special_conflict, zodiac_single, zodiac_two,
             special_zodiacs, strategy_specials, strategy_special_zodiacs,
             strategy_strong_special, strategy_strong_zodiac) = rec
            special_text = _fmt_num(special)
            trio_str = " ".join(_fmt_num(n) for n in predict_trio) if predict_trio else "无"
            defense_text = " ".join(_fmt_num(n) for n in special_defenses) if special_defenses else "无"
            strong_special_text = _fmt_num(strategy_strong_special) if strategy_strong_special is not None else "无"
            strong_zodiac_text = strategy_strong_zodiac if strategy_strong_zodiac else "无"
            special_zodiacs_text = "、".join(special_zodiacs) if special_zodiacs else "无"
            strategy_special_text = " ".join(_fmt_num(n) for n in strategy_specials) if strategy_specials else "无"
            strategy_zodiac_text = "、".join(strategy_special_zodiacs) if strategy_special_zodiacs else "无"

            all_specials = []
            for strategy in STRATEGY_IDS:
                run = conn.execute(
                    "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
                    (issue_no, strategy)
                ).fetchone()
                if run:
                    _, sp = get_picks_for_run(conn, run["id"])
                    if sp is not None:
                        all_specials.append(sp)
            unique_specials = []
            for sp in all_specials:
                if sp not in unique_specials:
                    unique_specials.append(sp)
            all_specials_str = " ".join(_fmt_num(n) for n in unique_specials) if unique_specials else "无"

            top_special_votes = get_top_special_votes(conn, issue_no, top_n=3)
            top_special_str = " ".join(_fmt_num(n) for n in top_special_votes) if top_special_votes else "无"

            stats_10 = get_review_stats(conn, window=10)
            confidence = max((float(s.get("hit1_rate") or 0.0) for s in stats_10), default=0.0) * 100.0
            max_miss = max((int(s.get("max_miss_streak", 0)) for s in stats_10), default=0)
            if confidence >= 80 and max_miss < 3:
                advice = "🔥 高信心：可适当加大投入"
            elif confidence >= 60:
                advice = "👍 中等信心：正常投入"
            else:
                advice = "⚠️ 低信心：建议减少投入或观望"

            zodiac_single_text = zodiac_single if zodiac_single else "数据不足"
            zodiac_two_text = "、".join(zodiac_two) if zodiac_two else "数据不足"
            conflict_tip = "（已避开主号冲突）" if special_conflict else ""

            content = (
                f"【新澳门·{issue_no}期推荐】\n"
                f"2生肖推荐：{zodiac_two_text}\n"
                f"1生肖推荐：{zodiac_single_text}\n"
                f"特别生肖推荐：{special_zodiacs_text}\n"
                f"特别号主推：{special_text}{conflict_tip}\n"
                f"特别号防守：{defense_text}\n"
                f"信心指数：{confidence:.0f}/100\n"
                f"建议投入：{advice}\n"
                f"六策略极强号：{strong_special_text}（{strong_zodiac_text}）\n"
                f"六策略特别号组：{strategy_special_text}\n"
                f"六策略生肖组：{strategy_zodiac_text}\n"
                f"特别号综合汇总（各策略去重）：{all_specials_str}\n"
                f"最终投票特别号（前三热门）：{top_special_str}\n"
                f"三中三预测（综合20码池+动态权重）：{trio_str}\n"
                f"详情请运行 python newmacau_marksix.py show"
            )
            send_pushplus_notification(f"新澳门预测 {issue_no}", content)


# ========== 命令行函数 ==========
def cmd_bootstrap(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        records = fetch_macau_records(timeout=args.api_timeout, retries=args.api_retries)
        total, inserted, updated = sync_from_records(conn, records, source="macau_api")
        print(f"自动执行轻量回测（最近{BACKTEST_ISSUES_DEFAULT}期）...")
        run_historical_backtest(conn, rebuild=True, max_issues=BACKTEST_ISSUES_DEFAULT)
        issue = generate_predictions(conn)
        print(f"Bootstrap done. total={total}, inserted={inserted}, updated={updated}, next_prediction={issue}")
    finally:
        conn.close()


def cmd_sync(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)

        records = fetch_records_with_fallback(
            limit=100,
            timeout=args.api_timeout,
            retries=args.api_retries,
        )
        if not records:
            print("[错误] 所有数据源均失败，请稍后重试")
            return

        total, inserted, updated = sync_from_records(conn, records, source="macau_api")
        mined_cfg = ensure_mined_pattern_config(conn, force=args.remine)
        reviewed = review_latest(conn)
        bt_issues, bt_runs = 0, 0

        if args.with_backtest:
            bt_issues, bt_runs = run_historical_backtest(conn, rebuild=False, max_issues=BACKTEST_ISSUES_DEFAULT)

        issue = generate_predictions(conn)
        patched = backfill_missing_special_picks(conn)

        cnt = conn.execute("SELECT COUNT(*) FROM special_picks_log").fetchone()[0]
        if cnt == 0 and total > 30:
            print("[自动] 检测到 special_picks_log 为空，开始回溯历史精选特别号...")
            backfill_special_picks_log(conn, max_issues=100)

        print(f"Sync done. total={total}, inserted={inserted}, updated={updated}, reviewed={reviewed}, next_prediction={issue}")
        print(f"Mined config: {json.dumps(mined_cfg, ensure_ascii=False)}")
        if bt_issues > 0:
            print(f"Backtest updated. issues={bt_issues}, strategy_runs={bt_runs}")
        if patched > 0:
            print(f"Patched missing special picks: {patched}")
    finally:
        conn.close()


def cmd_reset_and_auto(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        conn.execute("DELETE FROM prediction_picks")
        conn.execute("DELETE FROM prediction_pools")
        conn.execute("DELETE FROM prediction_runs")
        conn.execute("DELETE FROM strategy_performance")
        conn.execute("DELETE FROM special_picks_log")
        conn.execute("DELETE FROM model_state WHERE key = ?", (MINED_CONFIG_KEY,))
        conn.commit()

        optuna_path = SCRIPT_DIR / "optuna_macau_stable.db"
        if optuna_path.exists():
            os.remove(optuna_path)
            print(f"[重置] 已删除 {optuna_path}")

        records = fetch_records_with_fallback(limit=100, timeout=args.api_timeout, retries=args.api_retries)
        total, inserted, updated = sync_from_records(conn, records, source="auto_reset")
        mined_cfg = ensure_mined_pattern_config(conn, force=True)

        max_retries = 5
        trials = getattr(args, "trials", 1000)
        optimize_script = SCRIPT_DIR / "hyper_optimize_ultimate_target.py"
        debug_flag = ["--debug"] if getattr(args, "debug", False) else []

        for attempt in range(1, max_retries + 1):
            print(f"\n====== 第 {attempt} 次优化 (trials={trials}) ======")
            ret = subprocess.run(
                [sys.executable, str(optimize_script),
                 "--db", args.db,
                 "--recent", "120",
                 "--trials", str(trials)] + debug_flag,
                capture_output=False,
            )
            if ret.returncode == 0:
                print(f"🎉 第 {attempt} 次优化已达标！")
                break
            else:
                print(f"第 {attempt} 次优化未达标（退出码 {ret.returncode}），继续下一轮...")
                trials = int(trials * 1.3)

        run_historical_backtest(conn, rebuild=True, max_issues=120)
        generate_predictions(conn)
        print(f"Reset+auto done. total={total}, inserted={inserted}, updated={updated}")
        print(f"Mined config: {json.dumps(mined_cfg, ensure_ascii=False)}")
    finally:
        conn.close()


def cmd_sync_recent(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        records = fetch_recent_from_html()
        if not records:
            records = fetch_macau_recent_records(
                limit=args.limit,
                timeout=args.api_timeout,
                retries=args.api_retries,
            )
        total, inserted, updated = sync_from_records(conn, records, source="macau_api_recent")
        print(f"Recent sync done. limit={args.limit}, total={total}, inserted={inserted}, updated={updated}")
    finally:
        conn.close()


def cmd_predict(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        issue = generate_predictions(conn, issue_no=args.issue)
        patched = backfill_missing_special_picks(conn)
        print(f"Predictions generated for {issue}")
        if patched > 0:
            print(f"Patched missing special picks: {patched}")
    finally:
        conn.close()


def cmd_review(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        reviewed = review_issue(conn, args.issue) if args.issue else review_latest(conn)
        print(f"Reviewed runs: {reviewed}")
    finally:
        conn.close()


def cmd_show(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        backfill_missing_special_picks(conn)

        reviewed_count = conn.execute(
            "SELECT COUNT(*) FROM prediction_runs WHERE status='REVIEWED'"
        ).fetchone()[0]
        if reviewed_count < 10:
            print("检测到复盘数据不足，自动执行 sync --with-backtest ...")
            records = fetch_records_with_fallback(limit=100, timeout=args.api_timeout, retries=args.api_retries)
            if records:
                sync_from_records(conn, records, source="macau_api")
                run_historical_backtest(conn, rebuild=False, max_issues=100)
                generate_predictions(conn)
                print("自动同步与回测完成。")
            else:
                print("[错误] 自动同步失败，跳过回测与预测。")
                return

        # 自动检查并回溯缺失的精选特别号记录
        cnt = conn.execute("SELECT COUNT(*) FROM special_picks_log").fetchone()[0]
        if cnt == 0:
            print("[自动] 发现 special_picks_log 为空，开始回溯历史精选特别号...")
            backfill_special_picks_log(conn, max_issues=100)

        recent_online = None
        if records:
            recent_online = sorted(records, key=lambda r: (r.issue_no, r.draw_date))[-10:]
        print_dashboard(conn, recent_online=recent_online)
    finally:
        conn.close()

def cmd_backtest(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        mined_cfg = ensure_mined_pattern_config(conn, force=args.remine)
        issues, runs = run_historical_backtest(
            conn,
            min_history=args.min_history,
            rebuild=args.rebuild,
            progress_every=args.progress_every,
            max_issues=args.max_issues if hasattr(args, 'max_issues') else BACKTEST_ISSUES_DEFAULT,
        )
        print(f"Backtest done. issues={issues}, strategy_runs={runs}, rebuild={args.rebuild}")
        print(f"Mined config: {json.dumps(mined_cfg, ensure_ascii=False)}")
    finally:
        conn.close()


def cmd_mine(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        cfg = ensure_mined_pattern_config(conn, force=True)
        print(f"Mine done. config={json.dumps(cfg, ensure_ascii=False)}")
    finally:
        conn.close()


def cmd_backfill_special(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        count = backfill_special_picks_log(conn, max_issues=100)
        print(f"已回溯并写入 {count} 期精选特别号记录。")
    finally:
        conn.close()


# ========== 全自动优化命令 ==========
def evaluate_zodiac_performance(conn, params: dict, lookback: int = 20):
    import shutil
    backup_path = _BEST_PARAMS_PATH.with_suffix(".backup")
    if _BEST_PARAMS_PATH.exists():
        shutil.copy(_BEST_PARAMS_PATH, backup_path)
    try:
        with open(_BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False)
        single_rep = get_recent_single_zodiac_report(conn, lookback=lookback)
        two_rep = get_recent_two_zodiac_report(conn, lookback=lookback)
        three_rep = get_recent_three_zodiac_report(conn, lookback=lookback)
        four_rep = get_recent_four_zodiac_report(conn, lookback=lookback)
        hit_rates = {
            'single': single_rep['hit_rate'],
            'two': two_rep['hit_rate'],
            'three': three_rep['hit_rate'],
            'special': four_rep['hit_rate']
        }
        max_misses = {
            'single': single_rep['max_miss_streak'],
            'two': two_rep['max_miss_streak'],
            'three': three_rep['max_miss_streak'],
            'special': four_rep['max_miss_streak']
        }
        return hit_rates, max_misses
    finally:
        if backup_path.exists():
            shutil.copy(backup_path, _BEST_PARAMS_PATH)
            backup_path.unlink()


def auto_optimize_loop(conn, target_hit_rate=0.90, target_max_miss=1, 
                       timeout_hours=5, base_trials=200):
    try:
        import optuna
    except ImportError:
        print("❌ 请安装 optuna: pip install optuna")
        return None

    start_time = time.time()
    timeout_seconds = timeout_hours * 3600

    print(f"📥 同步最新100期开奖数据（超时限制：{timeout_hours}小时）...")
    records = fetch_records_with_fallback(limit=100)
    sync_from_records(conn, records, source="auto_optimize")
    total_rows = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
    print(f"✅ 当前数据库共 {total_rows} 期开奖记录")

    best_overall_params = None
    best_overall_score = -1
    round_num = 1
    n_trials = base_trials

    def get_params(trial, round_num):
        dynamic_max = lambda base, inc: min(base + inc * (round_num-1), base*2)
        params = {
            "single_strategy": trial.suggest_categorical("single_strategy",
                ["weighted", "pure_hot", "pure_cold", "hybrid", "hot_main_only", "last_special", "cold_hot_mix"]),
            "two_strategy": trial.suggest_categorical("two_strategy",
                ["hot_cold", "double_hot", "double_cold", "last2_specials", "hot_special_cold_main", "neighbor_pair"]),
            "four_strategy": trial.suggest_categorical("four_strategy",
                ["boosted", "momentum", "hybrid", "top4_freq", "cold_main_only"]),
            "special_strategy": trial.suggest_categorical("special_strategy",
                ["cold_neighbor", "tail_focus", "omission_only", "zone_bias", "neighbor_tail", "mixed_2cold1hot", "omit_break"]),
            "wsize": trial.suggest_int("wsize", 3, dynamic_max(15, 1)),
            "rec_w": trial.suggest_float("rec_w", 0.3, dynamic_max(1.6, 0.1)),
            "safe_th": trial.suggest_float("safe_th", 0.5, dynamic_max(3.0, 0.2)),
            "four_boost": trial.suggest_float("four_boost", 0.8, dynamic_max(3.0, 0.2)),
            "momentum_w": trial.suggest_float("momentum_w", 0.3, dynamic_max(2.0, 0.15)),
            "cold_threshold": trial.suggest_int("cold_threshold", 5, dynamic_max(25, 3)),
            "neighbor_1_bonus": trial.suggest_float("neighbor_1_bonus", 1.0, dynamic_max(12.0, 1.0)),
            "neighbor_2_bonus": trial.suggest_float("neighbor_2_bonus", 0.1, dynamic_max(4.0, 0.5)),
            "penalty_coeff": trial.suggest_float("penalty_coeff", 0.2, dynamic_max(1.8, 0.15)),
            "lgb_weight": trial.suggest_float("lgb_weight", 0.1, dynamic_max(1.5, 0.1)),
            "four_omit_boost": trial.suggest_float("four_omit_boost", 0.5, dynamic_max(6.0, 0.5)),
        }
        return params

    def objective(trial, round_num):
        params = get_params(trial, round_num)
        hit_rates, max_misses = evaluate_zodiac_performance(conn, params, lookback=20)
        score = hit_rates['single'] + hit_rates['two'] + hit_rates['three'] + hit_rates['special']
        if hit_rates['single'] < target_hit_rate or max_misses['single'] > target_max_miss:
            score *= 0.1
        if hit_rates['special'] < target_hit_rate or max_misses['special'] > target_max_miss:
            score *= 0.1
        if hit_rates['two'] < target_hit_rate or max_misses['two'] > target_max_miss:
            score *= 0.4
        if hit_rates['three'] < target_hit_rate or max_misses['three'] > target_max_miss:
            score *= 0.4
        if hit_rates['single'] >= 0.85 and hit_rates['special'] >= 0.85:
            score *= 1.5
        return score

    while (time.time() - start_time) < timeout_seconds:
        remaining_hours = (timeout_seconds - (time.time() - start_time)) / 3600
        print(f"\n{'='*50}\n第 {round_num} 轮优化 | 剩余时间: {remaining_hours:.1f}小时 | 本轮试验数: {n_trials}\n{'='*50}")

        study = optuna.create_study(direction="maximize", study_name=f"zodiac_opt_round{round_num}", load_if_exists=False)
        time_for_this_round = max(120, remaining_hours * 3600 - 600)
        study.optimize(lambda trial: objective(trial, round_num), 
                       n_trials=n_trials, 
                       timeout=time_for_this_round,
                       show_progress_bar=True)

        best_params = study.best_params
        best_params.setdefault("single_window", best_params.get("wsize", 6))
        best_params.setdefault("single_recency_w", best_params.get("rec_w", 0.7339))
        best_params.setdefault("single_safe_threshold", best_params.get("safe_th", 1.4589))

        hit_rates, max_misses = evaluate_zodiac_performance(conn, best_params, lookback=20)
        score = hit_rates['single'] + hit_rates['two'] + hit_rates['three'] + hit_rates['special']
        if score > best_overall_score:
            best_overall_score = score
            best_overall_params = best_params

        print(f"\n第 {round_num} 轮最佳表现:")
        print(f"  单肖: {hit_rates['single']:.3f} (连空 {max_misses['single']})")
        print(f"  双肖: {hit_rates['two']:.3f} (连空 {max_misses['two']})")
        print(f"  三肖: {hit_rates['three']:.3f} (连空 {max_misses['three']})")
        print(f"  特别肖: {hit_rates['special']:.3f} (连空 {max_misses['special']})")

        if (hit_rates['single'] >= target_hit_rate and max_misses['single'] <= target_max_miss and
            hit_rates['two'] >= target_hit_rate and max_misses['two'] <= target_max_miss and
            hit_rates['three'] >= target_hit_rate and max_misses['three'] <= target_max_miss and
            hit_rates['special'] >= target_hit_rate and max_misses['special'] <= target_max_miss):
            print("\n🎉 恭喜！所有指标均已达标！")
            with open(_BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
                json.dump(best_params, f, ensure_ascii=False, indent=2)
            return best_params

        n_trials += 200
        round_num += 1

        if remaining_hours < 1:
            print("⚠️ 剩余时间不足1小时，停止新试验，保存当前最佳参数。")
            break

    print(f"\n⏰ 优化时间达到 {timeout_hours} 小时限制，保存当前最佳参数。")
    with open(_BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_overall_params, f, ensure_ascii=False, indent=2)
    final_hit, final_miss = evaluate_zodiac_performance(conn, best_overall_params, lookback=20)
    print(f"最终表现: 单肖{final_hit['single']:.3f}(连空{final_miss['single']}) 双肖{final_hit['two']:.3f}(连空{final_miss['two']}) 三肖{final_hit['three']:.3f}(连空{final_miss['three']}) 特别肖{final_hit['special']:.3f}(连空{final_miss['special']})")
    return best_overall_params


def cmd_auto_optimize(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        best = auto_optimize_loop(
            conn,
            target_hit_rate=args.target_hit_rate,
            target_max_miss=args.target_max_miss,
            timeout_hours=args.timeout_hours,
            base_trials=args.base_trials
        )
        if best:
            run_historical_backtest(conn, rebuild=True, max_issues=120)
            generate_predictions(conn)
            print_dashboard(conn)
    finally:
        conn.close()


def cmd_check_data(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        rows = conn.execute(
            "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC"
        ).fetchall()
        if not rows:
            print("数据校验结果：数据库中没有开奖记录。")
            return

        issues = [str(r["issue_no"]) for r in rows]
        duplicate_issues = sorted({x for x in issues if issues.count(x) > 1})
        parsed_rows = []
        invalid_rows = []
        for r in rows:
            try:
                nums = json.loads(r["numbers_json"])
            except Exception:
                invalid_rows.append((r["issue_no"], "numbers_json 无法解析"))
                continue
            sp = r["special_number"]
            if not isinstance(nums, list) or len(nums) != 6:
                invalid_rows.append((r["issue_no"], f"主号数量异常: {len(nums) if isinstance(nums, list) else 'N/A'}"))
            bad_nums = [n for n in nums if not isinstance(n, int) or not (1 <= int(n) <= 49)]
            if bad_nums:
                invalid_rows.append((r["issue_no"], f"主号存在非法号码: {bad_nums}"))
            if not isinstance(sp, int) or not (1 <= int(sp) <= 49):
                invalid_rows.append((r["issue_no"], f"特别号非法: {sp}"))
            parsed_rows.append(r)

        issue_keys = [issue_sort_key(x) for x in issues if issue_sort_key(x) is not None]
        missing_issues = []
        if issue_keys:
            width = 3
            first_parsed = parse_issue(issues[0])
            if first_parsed:
                _, _, width = first_parsed
            year_s, seq, _ = parse_issue(issues[0]) or ("00", 0, width)
            current_year = int(year_s)
            current_seq = seq
            expected = set(issues)
            for _ in range(len(issues) + 5):
                current_seq += 1
                if current_seq > 366:
                    current_year += 1
                    current_seq = 1
                candidate = build_issue(str(current_year).zfill(len(year_s)), current_seq, width)
                if candidate not in expected and conn.execute("SELECT 1 FROM draws WHERE issue_no=?", (candidate,)).fetchone() is None:
                    missing_issues.append(candidate)
                    if len(missing_issues) >= 20:
                        break

        date_errors = []
        for r in rows:
            d = str(r["draw_date"])
            if _parse_date(d) is None:
                date_errors.append(r["issue_no"])

        print("数据校验结果")
        print(f"  - 总期数: {len(rows)}")
        print(f"  - 重复期号: {len(duplicate_issues)}")
        if duplicate_issues:
            print(f"    {', '.join(duplicate_issues[:10])}")
        print(f"  - 缺失期号(估算): {len(missing_issues)}")
        if missing_issues:
            print(f"    {', '.join(missing_issues[:10])}")
        print(f"  - 非法记录: {len(invalid_rows)}")
        if invalid_rows:
            for issue_no, msg in invalid_rows[:10]:
                print(f"    {issue_no}: {msg}")
        print(f"  - 日期异常: {len(date_errors)}")
        if date_errors:
            print(f"    {', '.join(date_errors[:10])}")

        if duplicate_issues or missing_issues or invalid_rows or date_errors:
            print("结论：当前数据存在需要清理或核验的问题。")
        else:
            print("结论：未发现明显数据结构异常。")
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="新澳门六合彩预测工具 - v5 稳定版（支持HTML抓取）")
    p.add_argument("--db", default=DB_PATH_DEFAULT, help=f"SQLite db path (default: {DB_PATH_DEFAULT})")
    p.add_argument("--update", action="store_true", help="Quick sync from API (same as sync)")
    p.add_argument("--remine", action="store_true", help="Re-mine pattern config before sync/backtest")
    p.add_argument("--tail-backtest", action="store_true", help="Run tail backtest and print report")
    p.add_argument("--api-timeout", type=int, default=API_TIMEOUT_DEFAULT, help="API timeout seconds per request")
    p.add_argument("--api-retries", type=int, default=API_RETRIES_DEFAULT, help="API retry attempts when network timeout/error occurs")
    p.add_argument("--require-continuity", action="store_true", default=True, help="Fail update when issue sequence has gaps")
    p.add_argument("--no-require-continuity", dest="require_continuity", action="store_false", help="Allow gaps")
    p.add_argument("--with-backtest", action="store_true", help=f"Run incremental backtest after sync (default last {BACKTEST_ISSUES_DEFAULT} issues)")
    sub = p.add_subparsers(dest="command", required=False)

    p_boot = sub.add_parser("bootstrap", help="Initial import from API and generate next issue predictions")
    p_boot.set_defaults(func=cmd_bootstrap)

    p_sync = sub.add_parser("sync", help="Sync draws from API, review latest, generate next prediction")
    p_sync.add_argument("--with-backtest", action="store_true", help=f"Run incremental backtest after sync (default last {BACKTEST_ISSUES_DEFAULT} issues)")
    p_sync.set_defaults(func=cmd_sync)

    p_recent = sub.add_parser("recent", help="Fetch and store only the latest N draws from API")
    p_recent.add_argument("--limit", type=int, default=200, help="Number of recent issues to fetch")
    p_recent.set_defaults(func=cmd_sync_recent)

    p_predict = sub.add_parser("predict", help="Generate predictions for next or specified issue")
    p_predict.add_argument("--issue", help="Target issue, e.g. 26/136")
    p_predict.set_defaults(func=cmd_predict)

    p_review = sub.add_parser("review", help="Review pending runs for latest or specified issue")
    p_review.add_argument("--issue", help="Issue to review, e.g. 26/136")
    p_review.set_defaults(func=cmd_review)

    p_show = sub.add_parser("show", help="Show local dashboard summary (auto-backfill missing special picks)")
    p_show.set_defaults(func=cmd_show)

    p_backtest = sub.add_parser("backtest", help="Run historical backtest for all draw issues")
    p_backtest.add_argument("--min-history", type=int, default=3, help="Min history window before first backtest issue")
    p_backtest.add_argument("--rebuild", action="store_true", help="Rebuild reviewed backtest runs from scratch")
    p_backtest.add_argument("--remine", action="store_true", help="Re-mine pattern config before backtest")
    p_backtest.add_argument("--max-issues", type=int, default=BACKTEST_ISSUES_DEFAULT, help="只回测最近 N 期（0=全部）")
    p_backtest.add_argument("--progress-every", type=int, default=20, help="Print backtest progress every N processed issues (0 to disable)")
    p_backtest.set_defaults(func=cmd_backtest)

    p_mine = sub.add_parser("mine", help="Mine best pattern parameters from history")
    p_mine.set_defaults(func=cmd_mine)

    p_backfill_special = sub.add_parser("backfill-special", help="回溯历史精选特别号记录")
    p_backfill_special.set_defaults(func=cmd_backfill_special)

    p_auto = sub.add_parser("auto-optimize", help="全自动优化生肖及特别号参数（目标90%命中率，最大连空1）")
    p_auto.add_argument("--target-hit-rate", type=float, default=0.90, help="目标命中率 (0~1)")
    p_auto.add_argument("--target-max-miss", type=int, default=1, help="目标最大连空")
    p_auto.add_argument("--timeout-hours", type=float, default=5, help="最大运行小时数（默认5）")
    p_auto.add_argument("--base-trials", type=int, default=200, help="首轮试验次数")
    p_auto.set_defaults(func=cmd_auto_optimize)

    p_reset = sub.add_parser("reset-and-auto", help="全自动重置→获取200期→优化近10期→预测")
    p_reset.add_argument("--trials", type=int, default=1000, help="Optuna trials count")
    p_reset.add_argument("--debug", action="store_true", help="打印每期调试信息")
    p_reset.set_defaults(func=cmd_reset_and_auto)

    p_check = sub.add_parser("check-data", help="校验数据库开奖记录是否完整且合法")
    p_check.set_defaults(func=cmd_check_data)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.update:
        cmd_sync(args)
        return
    if args.tail_backtest:
        conn = connect_db(args.db)
        try:
            init_db(conn)
            hit_rate, samples, max_miss = backtest_tail(conn)
            print(f"Tail backtest: hit_rate={hit_rate*100:.1f}% samples={samples} max_miss={max_miss}")
        finally:
            conn.close()
    if not args.command:
        parser.error("Please provide a subcommand, or use --update.")
    args.func(args)


if __name__ == "__main__":
    main()
