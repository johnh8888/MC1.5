#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import socket
import sqlite3
import sys
import time
from urllib.error import URLError
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = str(SCRIPT_DIR / "newmacau_marksix.db")
CSV_PATH_DEFAULT = str(SCRIPT_DIR / "NewMacau_Mark_Six.csv")

# 澳门数据源（使用 marksix6.net API 中的“新澳门彩”）
MACAU_API_URL = "https://marksix6.net/index.php?api=1"
API_TIMEOUT_DEFAULT = 20
API_RETRIES_DEFAULT = 4
API_RETRY_BACKOFF_SECONDS = 2.0

MINED_CONFIG_KEY = "mined_strategy_config_v1"
ALL_NUMBERS = list(range(1, 50))
BAYES_SPECIAL_WINDOW = 120  # 长窗口统一120期
BAYES_ALPHA = 3.0          # Dirichlet 平滑强度

# ==================== 【优化后常量】 ====================
FEATURE_WINDOW_DEFAULT = 10

STRATEGY_BASE_WINDOWS = {
    "hot_v1": 6,
    "momentum_v1": 7,
    "cold_rebound_v1": 13,
    "balanced_v1": 10,
    "pattern_mined_v1": 6,
    "ensemble_v2": 10,
}

WEIGHT_WINDOW_DEFAULT = 30
HEALTH_WINDOW_DEFAULT = 18
BACKTEST_ISSUES_DEFAULT = 120

# Ensemble v3.1 配置
ENSEMBLE_DIVERSITY_BONUS = 0.18

# 偏态检测阈值（已调整）
BIAS_THRESHOLD = 0.65
BIAS_ADJUSTMENT = 0.40
FORCED_BIAS_COEFFICIENT = 0.75

STRATEGY_LABELS = {
    "balanced_v1": "组合策略",
    "hot_v1": "热号策略",
    "cold_rebound_v1": "冷号回补",
    "momentum_v1": "近期动量",
    "ensemble_v2": "集成投票",
    "pattern_mined_v1": "规律挖掘",
}
STRATEGY_IDS = ["balanced_v1", "hot_v1", "cold_rebound_v1", "momentum_v1", "ensemble_v2", "pattern_mined_v1"]
SPECIAL_ANALYSIS_ORDER = ["pattern_mined_v1", "ensemble_v2", "momentum_v1", "cold_rebound_v1", "hot_v1", "balanced_v1"]

# 生肖映射（正确版本：1=马，2=蛇，3=龙，4=兔，5=虎，6=牛，7=鼠，8=猪，9=狗，10=鸡，11=猴，12=羊）
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

# PushPlus 配置
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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


def parse_draw_csv(csv_path: str) -> List[DrawRecord]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    records: List[DrawRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            issue_no = _pick(row, ["期号", "期數", "issueNo", "issue_no"])
            draw_date = _parse_date(_pick(row, ["日期", "date", "drawDate", "draw_date"]))
            special = _pick(row, ["特别号码", "特別號碼", "special", "specialNumber", "no7", "n7"])

            numbers = _parse_numbers(_pick(row, ["中奖号码", "中獎號碼", "numbers", "result"]))
            if len(numbers) != 6:
                split_keys = ["中奖号码 1", "中獎號碼 1", "1"], ["2"], ["3"], ["4"], ["5"], ["6"]
                split_nums: List[int] = []
                ok = True
                for key_group in split_keys:
                    value = _pick(row, list(key_group))
                    if not value:
                        ok = False
                        break
                    try:
                        n = int(value)
                    except ValueError:
                        ok = False
                        break
                    if not (1 <= n <= 49):
                        ok = False
                        break
                    split_nums.append(n)
                if ok:
                    numbers = split_nums

            try:
                special_n = int(special)
            except ValueError:
                continue

            if not issue_no or not draw_date:
                continue
            if len(numbers) != 6 or not (1 <= special_n <= 49):
                continue

            records.append(
                DrawRecord(
                    issue_no=issue_no,
                    draw_date=draw_date,
                    numbers=numbers,
                    special_number=special_n,
                )
            )

    records.sort(key=lambda r: (r.draw_date, r.issue_no))
    dedup: Dict[str, DrawRecord] = {}
    for r in records:
        dedup[r.issue_no] = r
    return sorted(dedup.values(), key=lambda r: (r.draw_date, r.issue_no))


def parse_draw_csv_text(csv_text: str) -> List[DrawRecord]:
    records: List[DrawRecord] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for raw in reader:
        row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
        issue_no = _pick(row, ["期号", "期數", "issueNo", "issue_no"])
        draw_date = _parse_date(_pick(row, ["日期", "date", "drawDate", "draw_date"]))
        special = _pick(row, ["特别号码", "特別號碼", "special", "specialNumber", "no7", "n7"])

        numbers = _parse_numbers(_pick(row, ["中奖号码", "中獎號碼", "numbers", "result"]))
        if len(numbers) != 6:
            split_keys = ["中奖号码 1", "中獎號碼 1", "1"], ["2"], ["3"], ["4"], ["5"], ["6"]
            split_nums: List[int] = []
            ok = True
            for key_group in split_keys:
                value = _pick(row, list(key_group))
                if not value:
                    ok = False
                    break
                try:
                    n = int(value)
                except ValueError:
                    ok = False
                    break
                if not (1 <= n <= 49):
                    ok = False
                    break
                split_nums.append(n)
            if ok:
                numbers = split_nums

        try:
            special_n = int(special)
        except ValueError:
            continue

        if not issue_no or not draw_date:
            continue
        if len(numbers) != 6 or not (1 <= special_n <= 49):
            continue

        records.append(
            DrawRecord(
                issue_no=issue_no,
                draw_date=draw_date,
                numbers=numbers,
                special_number=special_n,
            )
        )

    records.sort(key=lambda r: (r.draw_date, r.issue_no))
    dedup: Dict[str, DrawRecord] = {}
    for r in records:
        dedup[r.issue_no] = r
    return sorted(dedup.values(), key=lambda r: (r.draw_date, r.issue_no))


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
            match = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
            if not match:
                continue
            expect_raw = match.group(1)
            numbers_str = match.group(2)
            num_list = _parse_numbers(numbers_str)
            if len(num_list) < 7:
                continue
            main_numbers = num_list[:6]
            special = num_list[6]

            if len(expect_raw) >= 7:
                year = expect_raw[2:4]
                seq = str(int(expect_raw[4:]))
                issue_no = f"{year}/{seq.zfill(3)}"
            else:
                issue_no = expect_raw

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
                if len(expect_raw) >= 7:
                    year = expect_raw[2:4]
                    seq = str(int(expect_raw[4:]))
                    issue_no = f"{year}/{seq.zfill(3)}"
                else:
                    issue_no = expect_raw
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


def sync_from_csv(conn: sqlite3.Connection, csv_path: str, source: str = "local_csv") -> Tuple[int, int, int]:
    records = parse_draw_csv(csv_path)
    return sync_from_records(conn, records, source)


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
        "w_omit": 0.50,
        "w_mom": 0.20,
        "w_pair": 0.00,
        "w_zone": 0.10,
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

    w_freq = float(config.get("w_freq", 0.45))
    w_omit = float(config.get("w_omit", 0.35))
    w_mom = float(config.get("w_mom", 0.20))
    w_pair = float(config.get("w_pair", 0.00))
    w_zone = float(config.get("w_zone", 0.00))

    scores: Dict[int, float] = {}
    for n in ALL_NUMBERS:
        scores[n] = (
            freq[n] * w_freq
            + omission[n] * w_omit
            + momentum[n] * w_mom
            + pair[n] * w_pair
            + zone[n] * w_zone
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

    # 冷号回补特殊处理：统计长期遗漏号码数量
    if strategy == "cold_rebound_v1":
        rows = conn.execute(
            "SELECT numbers_json FROM draws ORDER BY draw_date DESC LIMIT 60"
        ).fetchall()
        all_nums = []
        for r in rows:
            all_nums.extend(json.loads(r["numbers_json"]))
        freq = Counter(all_nums)
        cold_count = sum(1 for n in ALL_NUMBERS if freq.get(n, 0) == 0)
        if cold_count >= 5:   # 遗漏≥60期的号码超过5个
            return min(20, base + 8)   # 大幅扩大窗口

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
    """强制偏态模式：固定偏态系数 0.75"""
    return 0.75, {
        "forced": True,
        "zone_bias": 0.75,
        "parity_bias": 0.70,
        "hot_cold_bias": 0.70,
        "zone_dist": [0]*5,
        "odd_ratio": 0.5
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
# ===================== Bayesian Likelihood Builders =====================
def _build_omission_cond_counts(conn, window=120, alpha=3.0):
    """
    构建遗漏条件计数表 (omit >= k 时的特别号分布)
    """
    rows = conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date ASC LIMIT ?",
        (window,)
    ).fetchall()
    specials = [int(r[0]) for r in rows]
    
    # 计算每期的遗漏值（特别号自身）
    last_seen = {}
    omit_at = []  # omit_at[i] = 第 i 期特别号的遗漏值
    for i, sp in enumerate(specials[:-1]):  # 最后一期作未来用
        omit = i - last_seen[sp] if sp in last_seen else window
        omit_at.append(omit)
        last_seen[sp] = i
    
    # 计数: cond_counts[omit_val][next_sp]
    cond_counts = defaultdict(Counter)
    cond_totals = defaultdict(int)
    for i, omit in enumerate(omit_at):
        next_sp = specials[i+1]
        cond_counts[omit][next_sp] += 1
        cond_totals[omit] += 1
    
    # 转为累积 >= k 的计数
    cum_counts = {}
    cum_totals = {}
    running = Counter()
    run_total = 0
    for k in range(window, 0, -1):
        if k in cond_counts:
            running += cond_counts[k]
            run_total += cond_totals[k]
        cum_counts[k] = Counter(running)
        cum_totals[k] = run_total
    # 大于 window 的遗漏统一用 max_omit
    max_omit = window
    cum_counts[max_omit] = Counter(running)
    cum_totals[max_omit] = run_total
    return cum_counts, cum_totals


def _build_association_cond_counts(conn, window=120, alpha=3.0):
    """
    构建主号关联条件计数表（尾数、邻号、生肖）
    """
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date ASC LIMIT ?",
        (window,)
    ).fetchall()
    tail_cond = defaultdict(Counter)
    neigh_cond = defaultdict(Counter)  # key=(main_number, delta)
    zodiac_cond = defaultdict(Counter)
    for r in rows[:-1]:
        mains = json.loads(r["numbers_json"])
        sp = int(r["special_number"])
        # 尾数
        for m in mains:
            tail_cond[m % 10][sp] += 1
        # 邻号
        for m in mains:
            for d in [-2, -1, 1, 2]:
                neigh_cond[(m, d)][sp] += 1
        # 生肖
        for z in {get_zodiac_by_number(m) for m in mains}:
            zodiac_cond[z][sp] += 1
    return tail_cond, neigh_cond, zodiac_cond


# ===================== Bayesian Likelihood Builders =====================
def _build_omission_cond_counts(conn, window=120, alpha=3.0):
    """
    构建遗漏条件计数表 (omit >= k 时的特别号分布)
    """
    rows = conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date ASC LIMIT ?",
        (window,)
    ).fetchall()
    specials = [int(r[0]) for r in rows]

    last_seen = {}
    omit_at = []
    for i, sp in enumerate(specials[:-1]):
        omit = i - last_seen[sp] if sp in last_seen else window
        omit_at.append(omit)
        last_seen[sp] = i

    from collections import defaultdict
    cond_counts = defaultdict(Counter)
    cond_totals = defaultdict(int)
    for i, omit in enumerate(omit_at):
        next_sp = specials[i + 1]
        cond_counts[omit][next_sp] += 1
        cond_totals[omit] += 1

    cum_counts = {}
    cum_totals = {}
    running = Counter()
    run_total = 0
    for k in range(window, 0, -1):
        if k in cond_counts:
            running += cond_counts[k]
            run_total += cond_totals[k]
        cum_counts[k] = Counter(running)
        cum_totals[k] = run_total
    cum_counts[window] = Counter(running)
    cum_totals[window] = run_total
    return cum_counts, cum_totals


def _build_association_cond_counts(conn, window=120, alpha=3.0):
    """
    构建主号关联条件计数表（尾数、邻号、生肖）
    """
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date ASC LIMIT ?",
        (window,)
    ).fetchall()
    tail_cond = defaultdict(Counter)
    neigh_cond = defaultdict(Counter)
    zodiac_cond = defaultdict(Counter)
    for r in rows[:-1]:
        mains = json.loads(r["numbers_json"])
        sp = int(r["special_number"])
        for m in mains:
            tail_cond[m % 10][sp] += 1
            for d in [-2, -1, 1, 2]:
                neigh_cond[(m, d)][sp] += 1
        for z in {get_zodiac_by_number(m) for m in mains}:
            zodiac_cond[z][sp] += 1
    return tail_cond, neigh_cond, zodiac_cond


def _generate_special_number_bayesian_v6(
    conn: sqlite3.Connection,
    main_pool: List[int],
    issue_no: str,
    alpha: float = 3.0,
    long_window: int = 120,
) -> Tuple[int, float, List[int]]:
    draw_sig = conn.execute("SELECT COUNT(*) AS c, MAX(draw_date) AS d FROM draws").fetchone()
    cache_key = (int(draw_sig["c"] or 0), str(draw_sig["d"] or ""), int(long_window), float(alpha))

    omit_cache_key = getattr(_generate_special_number_bayesian_v6, "omit_cache_key", None)
    if omit_cache_key != cache_key:
        _generate_special_number_bayesian_v6.omit_cache = _build_omission_cond_counts(conn, window=long_window, alpha=alpha)
        _generate_special_number_bayesian_v6.omit_cache_key = cache_key
    omit_cum_counts, omit_cum_totals = _generate_special_number_bayesian_v6.omit_cache

    assoc_cache_key = getattr(_generate_special_number_bayesian_v6, "assoc_cache_key", None)
    if assoc_cache_key != cache_key:
        _generate_special_number_bayesian_v6.assoc_cache = _build_association_cond_counts(conn, window=long_window, alpha=alpha)
        _generate_special_number_bayesian_v6.assoc_cache_key = cache_key
    tail_cond, neigh_cond, zodiac_cond = _generate_special_number_bayesian_v6.assoc_cache

    all_specials_seq = [int(r["special_number"]) for r in conn.execute(
        f"SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT {long_window}"
    ).fetchall()]
    omission_raw = {n: long_window for n in ALL_NUMBERS}
    for idx, sp in enumerate(all_specials_seq):
        if omission_raw[sp] == long_window:
            omission_raw[sp] = idx + 1

    special_votes = []
    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (issue_no, strategy)
        ).fetchone()
        if run:
            _, sp = get_picks_for_run(conn, run["id"])
            if sp is not None:
                special_votes.append(sp)
    vote_counter = Counter(special_votes)

    recent_specials = all_specials_seq[:20]
    recent_weights = {sp: 0.0 for sp in ALL_NUMBERS}
    for idx, sp in enumerate(recent_specials):
        recent_weights[sp] += 1.0 / (1.0 + idx * 0.35)
    recent_weight_total = sum(recent_weights.values()) or 1.0

    main_set = set(main_pool)
    main_tails = {m % 10 for m in main_pool}
    main_zodiacs = {get_zodiac_by_number(m) for m in main_pool}

    scores = {}
    for n in ALL_NUMBERS:
        if n in main_set:
            continue

        cur_omit = min(omission_raw.get(n, long_window), long_window)
        o_cnt = omit_cum_counts.get(cur_omit, Counter()).get(n, 0)
        o_total = omit_cum_totals.get(cur_omit, 0)
        ll_omit = math.log((o_cnt + alpha) / (o_total + alpha * 49) + 1e-12) if o_total > 0 else 0.0

        ll_tail = 0.0
        for t in main_tails:
            t_counter = tail_cond.get(t, Counter())
            t_total = sum(t_counter.values())
            if t_total > 0:
                t_cnt = t_counter.get(n, 0)
                ll_tail += math.log((t_cnt + alpha) / (t_total + alpha * 49) + 1e-12)

        ll_neigh = 0.0
        for m in main_pool:
            for d in [-2, -1, 1, 2]:
                n_counter = neigh_cond.get((m, d), Counter())
                n_total = sum(n_counter.values())
                if n_total > 0:
                    nc = n_counter.get(n, 0)
                    ll_neigh += math.log((nc + alpha) / (n_total + alpha * 49) + 1e-12)

        ll_zodiac = 0.0
        for z in main_zodiacs:
            z_counter = zodiac_cond.get(z, Counter())
            z_total = sum(z_counter.values())
            if z_total > 0:
                zc = z_counter.get(n, 0)
                ll_zodiac += math.log((zc + alpha) / (z_total + alpha * 49) + 1e-12)

        ll_assoc = (ll_tail * 1.5 + ll_neigh * 2.0 + ll_zodiac * 0.8) / 3.0

        vote_support = vote_counter.get(n, 0)
        ll_freq = math.log((vote_support + 1.0) / (len(special_votes) + 49.0) + 1e-12)

        recency_support = recent_weights.get(n, 0.0) / recent_weight_total
        ll_mom = math.log((recency_support + 1e-12) / (1.0 / 49.0) + 1e-12)

        log_prior = math.log(1.0 / 49.0 + 1e-12)
        scores[n] = (
            log_prior * 1.0
            + ll_omit * 2.0
            + ll_assoc * 2.6
            + ll_freq * 0.45
            + ll_mom * 0.25
        )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best = ranked[0][0]
    confidence = min(1.0, (ranked[0][1] - ranked[-1][1]) / 15.0) if len(ranked) > 1 else 1.0
    defenses = [n for n, _ in ranked[1:4]]
    return best, round(confidence, 3), defenses


def _generate_special_number_v4(
    conn: sqlite3.Connection,
    main_pool: List[int],
    issue_no: str
) -> Tuple[int, float, List[int]]:
    return _generate_special_number_bayesian_v6(conn, main_pool, issue_no)
    special_votes = []
    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (issue_no, strategy)
        ).fetchone()
        if run:
            _, sp = get_picks_for_run(conn, run["id"])
            if sp is not None:
                special_votes.append(sp)
    vote_counter = Counter(special_votes)

    recent_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 60"
    ).fetchall()]

    omission = {n: 60 for n in ALL_NUMBERS}
    for i, num in enumerate(recent_specials):
        omission[num] = min(omission.get(num, 60), i + 1)

    zodiac_cycle = [get_zodiac_by_number(sp) for sp in recent_specials[:24]]
    if len(zodiac_cycle) >= 6:
        recent_zodiacs = zodiac_cycle[:6]
        zodiac_counter = Counter(recent_zodiacs)
        least_zodiac = min(zodiac_counter.keys(), key=lambda z: zodiac_counter[z])
        predicted_zodiac_numbers = ZODIAC_MAP.get(least_zodiac, [1, 13, 25, 37, 49])
    else:
        predicted_zodiac_numbers = ALL_NUMBERS

    tail_counter = Counter([n % 10 for n in recent_specials[:20]])
    coldest_tail = min(tail_counter.keys(), key=lambda t: tail_counter[t]) if tail_counter else 0

    main_zones_set = {(m - 1) // 10 for m in main_pool}
    main_set = set(main_pool)

    # === 新增：主号生肖缺失补偿分析 ===
    recent_main_zodiacs = []
    for row in conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC LIMIT 2"
    ).fetchall():
        for n in json.loads(row["numbers_json"]):
            recent_main_zodiacs.append(get_zodiac_by_number(n))
    missing_zodiacs = set(ZODIAC_MAP.keys()) - set(recent_main_zodiacs[-12:]) if len(recent_main_zodiacs) >= 12 else set()
    # ===================================

    scores = {}
    for n in ALL_NUMBERS:
        if n in main_set:
            continue
        score = 0.0
        score += vote_counter.get(n, 0) * 6.0
        omission_score = (60 - omission.get(n, 60)) / 60.0
        if omission.get(n, 60) > 12:
            score += omission_score * 4.0
        else:
            score += omission_score * 1.0
        if n in recent_specials[:2]:
            score *= 0.2
        elif n in recent_specials[2:5]:
            score *= 0.6
        if n in predicted_zodiac_numbers:
            score += 1.8
        if n % 10 == coldest_tail:
            score += 1.2
        for mn in main_pool:
            diff = abs(n - mn)
            if diff == 1:
                score += 2.0
            elif diff == 2:
                score += 1.5
            elif diff == 3:
                score += 1.0
            elif n % 10 == mn % 10 and n != mn:
                score += 0.8
        n_zone = (n - 1) // 10
        if n_zone in main_zones_set:
            score += 1.2
        recent_parity = [sp % 2 for sp in recent_specials[:5]]
        if len(recent_parity) >= 3:
            odd_ratio = sum(recent_parity) / len(recent_parity)
            if odd_ratio > 0.6 and n % 2 == 0:
                score += 1.0
            elif odd_ratio < 0.4 and n % 2 == 1:
                score += 1.0

        # === 新增：主号生肖缺失补偿加分 ===
        if missing_zodiacs and get_zodiac_by_number(n) in missing_zodiacs:
            score += 3.0
        # ================================

        scores[n] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best = ranked[0][0]
    confidence = min(1.0, ranked[0][1] / 18)
    defenses = []
    for n, s in ranked[1:]:
        if n not in main_set and n != best:
            defenses.append(n)
        if len(defenses) >= 3:
            break
    while len(defenses) < 3:
        for n, s in ranked:
            if n not in defenses and n != best and n not in main_set:
                defenses.append(n)
                break

    return best, round(confidence, 3), defenses[:3]


# ========== 三中三优化版 v2 ==========
def get_trio_from_merged_pool20_v2(conn: sqlite3.Connection, issue_no: str) -> List[int]:
    _, _, _, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if not pool20 or len(pool20) < 3:
        return [1, 2, 3]
    all_pools = []
    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (issue_no, strategy)
        ).fetchone()
        if run:
            p20 = get_pool_numbers_for_run(conn, run["id"], 20)
            p20_filtered = [n for n in p20 if n in pool20]
            all_pools.extend(p20_filtered)
    if len(all_pools) < 3:
        return pool20[:3]
    appearance_count = Counter(all_pools)
    diff_numbers = [n for n, c in appearance_count.items() if 1 <= c <= 2 and n in pool20]
    if len(diff_numbers) < 6:
        diff_numbers = [n for n, c in appearance_count.items() if c <= 3 and n in pool20]
    if len(diff_numbers) < 3:
        diff_numbers = pool20[:15]
    draws = load_recent_draws(conn, FEATURE_WINDOW_DEFAULT)
    if len(draws) < 3:
        return diff_numbers[:3]
    momentum = _momentum_map(draws)
    freq = _freq_map(draws)
    omission = _omission_map(draws)
    momentum_norm = _normalize(momentum)
    freq_norm = _normalize(freq)
    omission_norm = _normalize(omission)
    w_mom, w_hot, w_cold = get_trio_weights(conn, window=WEIGHT_WINDOW_DEFAULT)
    scores = {}
    for n in diff_numbers[:15]:
        score = (w_mom * momentum_norm.get(n, 0) +
                 w_hot * freq_norm.get(n, 0) +
                 w_cold * omission_norm.get(n, 0))
        score += (6 - appearance_count.get(n, 3)) * 0.15
        scores[n] = score
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [n for n, _ in sorted_nums[:10]]

    def is_valid(trio):
        odd_cnt = sum(1 for x in trio if x % 2 == 1)
        total = sum(trio)
        return 1 <= odd_cnt <= 2 and 80 <= total <= 130

    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            for k in range(j+1, len(candidates)):
                trio = (candidates[i], candidates[j], candidates[k])
                if is_valid(trio):
                    return list(trio)
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            for k in range(j+1, len(candidates)):
                trio = (candidates[i], candidates[j], candidates[k])
                odd_cnt = sum(1 for x in trio if x % 2 == 1)
                if 1 <= odd_cnt <= 2:
                    return list(trio)
    return candidates[:3] if len(candidates) >= 3 else pool20[:3]


def _ensemble_strategy_v3_1(
    draws: List[List[int]],
    mined_config: Optional[Dict[str, float]],
    strategy_weights: Dict[str, float],
    conn: sqlite3.Connection,
    issue_no: str
) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:
    sub_strategies = ["hot_v1", "cold_rebound_v1", "momentum_v1", "balanced_v1", "pattern_mined_v1"]
    score_maps = []
    sub_picks = {}

    bias_score, _ = detect_bias(conn, window=10)
    adjusted_weights = adjust_weights_for_bias(strategy_weights, bias_score)

    if bias_score > BIAS_THRESHOLD:
        print(f"[集成策略] 偏态模式激活，偏态系数={bias_score:.2f}", flush=True)
        cold_weight = adjusted_weights.get("cold_rebound_v1", 0.0)
        print(f"   -> 冷号回补当前权重: {cold_weight:.3f}", flush=True)
    else:
        print(f"[集成策略] 正常模式，偏态系数={bias_score:.2f}", flush=True)

    for sub in sub_strategies:
        win_size = get_adaptive_strategy_window(sub, conn)
        sub_draws = draws[:win_size] if len(draws) > win_size else draws

        if sub == "pattern_mined_v1":
            cfg = mined_config or _default_mined_config()
            cfg["window"] = float(win_size)
            _, _, _, score_map = _apply_weight_config(sub_draws, cfg, "规律挖掘")
        else:
            config = {"window": float(win_size)}
            if sub == "hot_v1":
                config.update({"w_freq": 0.78, "w_omit": 0.05, "w_mom": 0.17})
            elif sub == "cold_rebound_v1":
                config.update({"w_freq": 0.05, "w_omit": 0.68, "w_mom": 0.27})
            elif sub == "momentum_v1":
                config.update({"w_freq": 0.12, "w_omit": 0.05, "w_mom": 0.83})
            else:
                config.update({"w_freq": 0.40, "w_omit": 0.30, "w_mom": 0.20})
            _, _, _, score_map = _apply_weight_config(sub_draws, config, STRATEGY_LABELS.get(sub, sub))

        score_maps.append(score_map)
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        sub_picks[sub] = [n for n, _ in ranked[:6]]

    votes = {n: 0.0 for n in ALL_NUMBERS}
    for idx, sub in enumerate(sub_strategies):
        w = adjusted_weights.get(sub, 0.2)
        ranked = sorted(score_maps[idx].items(), key=lambda x: x[1], reverse=True)
        for rank, (n, _) in enumerate(ranked):
            votes[n] += w * (49 - rank)

    # === 新增：强化冷号回补策略的贡献 ===
    cold_picks = sub_picks.get("cold_rebound_v1", [])
    for idx, n in enumerate(cold_picks):
        votes[n] += 0.8 * (6 - idx)
    # ===================================

    for n in ALL_NUMBERS:
        appear = sum(1 for p in sub_picks.values() if n in p)
        votes[n] += (6 - appear) * ENSEMBLE_DIVERSITY_BONUS * 1.2

    voted = _normalize(votes)
    main_picked = _pick_top_six(voted, "集成投票v3.1")

    main6 = [n for n, _, _, _ in main_picked]
    special_number, confidence, _ = _generate_special_number_bayesian_v6(conn, main6, issue_no)

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
            {"window": float(window_size), "w_freq": 0.78, "w_omit": 0.05, "w_mom": 0.17},
            "热号策略"
        )
    elif strategy == "cold_rebound_v1":
        return _apply_weight_config(
            strategy_draws,
            {"window": float(window_size), "w_freq": 0.05, "w_omit": 0.68, "w_mom": 0.27},
            "冷号回补"
        )
    elif strategy == "momentum_v1":
        return _apply_weight_config(
            strategy_draws,
            {"window": float(window_size), "w_freq": 0.12, "w_omit": 0.05, "w_mom": 0.83},
            "近期动量"
        )
    elif strategy == "balanced_v1":
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


def get_review_stats(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
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
        """
    ).fetchall()


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
def get_strategy_weights(conn: sqlite3.Connection, window: int = WEIGHT_WINDOW_DEFAULT) -> Dict[str, float]:
    rows = conn.execute("""
        SELECT strategy, AVG(main_hit_count) as avg_hit
        FROM strategy_performance
        WHERE issue_no IN (
            SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?
        )
        GROUP BY strategy
    """, (window,)).fetchall()

    baseline = 0.6
    weights = {s: baseline for s in STRATEGY_IDS}
    protection_msgs: List[str] = []

    for r in rows:
        strategy = str(r["strategy"])
        avg_hit = float(r["avg_hit"] or 0.0)
        if strategy in weights:
            weights[strategy] = max(avg_hit, baseline)

    health = get_strategy_health(conn, window=HEALTH_WINDOW_DEFAULT)
    for strategy, h in health.items():
        if strategy not in weights:
            continue
        recent_avg = float(h.get("recent_avg_hit", 0.0))
        hit1_rate = float(h.get("hit1_rate", 0.0))
        cold_streak = int(h.get("cold_streak", 0))

        shrink = 1.0
        if recent_avg < 0.7:
            shrink *= 0.90 ** ((0.7 - recent_avg) * 8)
        if hit1_rate < 0.52:
            shrink *= 0.87
        if cold_streak >= 3:
            shrink *= 0.72

        if strategy == "pattern_mined_v1" and (cold_streak >= 2 or recent_avg < 0.6):
            shrink *= 0.48
            protection_msgs.append(f"[保护] 规律挖掘连挂 {cold_streak} 期，权重大幅下调")

        weights[strategy] = max(0.08, weights[strategy] * shrink)

    total = sum(weights.values())
    global _PROTECTION_PRINT_COUNTER
    for msg in protection_msgs:
        if msg not in _WEIGHT_PROTECTION_PRINTED:
            print(msg, flush=True)
            _WEIGHT_PROTECTION_PRINTED.add(msg)
    if protection_msgs:
        _PROTECTION_PRINT_COUNTER += 1
        if _PROTECTION_PRINT_COUNTER % 20 == 0:
            print(f"[保护] 当前规律挖掘/冷号回补仍处于权重保护中 (已持续{_PROTECTION_PRINT_COUNTER}期)", flush=True)
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
def get_zodiac_by_number(number: int) -> str:
    for zodiac, nums in ZODIAC_MAP.items():
        if number in nums:
            return zodiac
    return "马"


def _get_previous_issue(conn: sqlite3.Connection, current_issue: str) -> Optional[str]:
    """获取当前期号的上一期"""
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
    """检查指定期号的双生肖推荐是否命中"""
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

    # 重新生成该期的双生肖推荐（与当前逻辑一致，但不包含上期补偿以避免递归）
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
    picks = [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else ["马", "蛇"]

    return any(z in winning_zodiacs for z in picks)


def _zodiac_omission_map(rows: Sequence[sqlite3.Row]) -> Dict[str, int]:
    """计算每个生肖最近一次出现的期数距离（遗漏值）"""
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
        if omit >= 6:
            zodiac_scores[z] += min(3.0, omit / 4.0)
        elif omit >= 3:
            zodiac_scores[z] += omit / 6.0
    return zodiac_scores


def get_two_zodiac_picks(conn: sqlite3.Connection, issue_no: str, window: int = 16) -> List[str]:
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (window,),
    ).fetchall()
    if not rows:
        return ["马", "蛇"]

    # 基础生肖得分
    zodiac_scores = _build_zodiac_scores_from_rows(rows, decay=0.08)
    omission_map = _zodiac_omission_map(rows)

    # 遗漏保护：阈值8期
    force_include = []
    for z, omit in omission_map.items():
        if omit >= 8:
            force_include.append(z)

    # 结合20码池
    _, _, _, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if pool20:
        pool_zodiacs = [get_zodiac_by_number(n) for n in pool20]
        for z, cnt in Counter(pool_zodiacs).items():
            zodiac_scores[z] += cnt * 0.6

    # 特别号投票加分
    top_special_votes = get_top_special_votes(conn, issue_no, top_n=3)
    if top_special_votes:
        for sp in top_special_votes:
            zodiac_scores[get_zodiac_by_number(sp)] += 1.5

    # ---------- 上期未命中补偿 ----------
    prev_issue = _get_previous_issue(conn, issue_no)
    prev_hit = False
    if prev_issue:
        prev_hit = _check_two_zodiac_hit(conn, prev_issue)

    # ===== 绝对兜底：上期未命中则直接返回上期最热两个生肖 =====
    if not prev_hit and prev_issue:
        prev_draw = conn.execute(
            "SELECT numbers_json, special_number FROM draws WHERE issue_no = ?",
            (prev_issue,)
        ).fetchone()
        if prev_draw:
            prev_zodiacs = []
            for n in json.loads(prev_draw["numbers_json"]):
                prev_zodiacs.append(get_zodiac_by_number(n))
            prev_zodiacs.append(get_zodiac_by_number(prev_draw["special_number"]))
            hot_two = [z for z, _ in Counter(prev_zodiacs).most_common(2)]
            if len(hot_two) >= 2:
                return hot_two[:2]

    # 正常流程
    ranked = sorted(zodiac_scores.items(), key=lambda x: (-x[1], x[0]))
    picks = []
    for z in force_include:
        if z not in picks:
            picks.append(z)
    for z, _ in ranked:
        if len(picks) >= 2:
            break
        if z not in picks:
            picks.append(z)

    # 兜底：如果仍未满两个，用最高分补齐
    if len(picks) < 2:
        for z, _ in ranked:
            if z not in picks:
                picks.append(z)
            if len(picks) == 2:
                break

    return picks[:2]


def get_single_zodiac_pick(conn: sqlite3.Connection, issue_no: str, window: int = 14) -> str:
    two_zodiac = get_two_zodiac_picks(conn, issue_no, window)
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (window,)
    ).fetchall()
    if not rows:
        return two_zodiac[0] if two_zodiac else "马"

    # 更短窗口 + 更激进的 decay
    zodiac_scores = _build_zodiac_scores_from_rows(rows, decay=0.05)

    # 动态遗漏加分：遗漏期数直接乘以系数
    omission_map = _zodiac_omission_map(rows)
    for z in zodiac_scores:
        omit = omission_map.get(z, len(rows))
        zodiac_scores[z] += min(5.0, omit * 0.8)   # 遗漏20期加16分，上限5分

    # 明确倾向最冷生肖
    coldest_zodiac = max(omission_map.keys(), key=lambda z: omission_map[z])
    zodiac_scores[coldest_zodiac] += 5.0   # 强化冷回补

    # 结合20码池（权重提升）
    _, _, _, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if pool20:
        pool_zodiacs = [get_zodiac_by_number(n) for n in pool20]
        for z, cnt in Counter(pool_zodiacs).items():
            zodiac_scores[z] += cnt * 0.8   # 原0.5 → 0.8

    # 特别号投票加分（权重提升）
    top_special_votes = get_top_special_votes(conn, issue_no, top_n=3)
    if top_special_votes:
        for sp in top_special_votes:
            zodiac_scores[get_zodiac_by_number(sp)] += 2.5   # 原1.8 → 2.5

    # 减弱近期重复惩罚
    recent_special_zodiacs = [get_zodiac_by_number(int(r["special_number"])) for r in rows[:3]]
    for z in recent_special_zodiacs:
        zodiac_scores[z] -= 0.1   # 原-0.3 → -0.1

    # 倾向双生肖推荐（绑定加强）
    for z in two_zodiac:
        zodiac_scores[z] += 4.0   # 原3.0 → 4.0

    ranked = sorted(zodiac_scores.items(), key=lambda x: (-x[1], x[0]))
    for candidate, _ in ranked:
        if candidate in two_zodiac:
            return candidate
    return ranked[0][0]


def get_hot_cold_zodiacs(conn: sqlite3.Connection, window: int = 12, top_n: int = 3) -> Tuple[List[str], List[str]]:
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (window,)
    ).fetchall()
    if len(rows) < window:
        default = ["马", "蛇", "龙", "兔", "虎", "牛"]
        return default[:top_n], default[-top_n:]
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


def _get_two_zodiac_from_history_rows(rows: Sequence[sqlite3.Row]) -> List[str]:
    if not rows:
        return ["马", "蛇"]
    zodiac_scores = _build_zodiac_scores_from_rows(rows, decay=0.10)
    recent_special_zodiacs = [get_zodiac_by_number(int(r["special_number"])) for r in rows[:3]]
    for z in recent_special_zodiacs:
        zodiac_scores[z] -= 0.2
    ranked = sorted(zodiac_scores.items(), key=lambda x: (-x[1], x[0]))
    return [ranked[0][0], ranked[1][0]] if len(ranked) >= 2 else ["马", "蛇"]


def _get_single_zodiac_from_history_rows(rows: Sequence[sqlite3.Row]) -> str:
    two_zodiac = _get_two_zodiac_from_history_rows(rows)
    if not rows:
        return two_zodiac[0] if two_zodiac else "马"

    zodiac_scores = _build_zodiac_scores_from_rows(rows, decay=0.12)
    recent_zodiacs = [get_zodiac_by_number(int(r["special_number"])) for r in rows[:12]]
    zodiac_counter = Counter(recent_zodiacs)
    if zodiac_counter:
        coldest = min(zodiac_counter.keys(), key=lambda z: zodiac_counter[z])
        zodiac_scores[coldest] += 4.0

    recent_special_zodiacs = [get_zodiac_by_number(int(r["special_number"])) for r in rows[:3]]
    for z in recent_special_zodiacs:
        zodiac_scores[z] -= 0.4

    for z in two_zodiac:
        zodiac_scores[z] += 3.0

    ranked = sorted(zodiac_scores.items(), key=lambda x: (-x[1], x[0]))
    for candidate, _ in ranked:
        if candidate in two_zodiac:
            return candidate
    return ranked[0][0]


def get_recent_single_zodiac_report(
    conn: sqlite3.Connection,
    lookback: int = 20,
    history_window: int = 14,
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
        pick = _get_single_zodiac_from_history_rows(history_rows)
        win_main = json.loads(rows[i]["numbers_json"])
        win_special = int(rows[i]["special_number"])
        winning_zodiacs = {get_zodiac_by_number(int(n)) for n in win_main}
        winning_zodiacs.add(get_zodiac_by_number(win_special))
        hit = 1 if pick in winning_zodiacs else 0
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


def get_recent_two_zodiac_report(
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
        picks = _get_two_zodiac_from_history_rows(history_rows)
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


# ========== 特别号投票 ==========
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


def get_special_recommendation(conn: sqlite3.Connection, issue_no: str, main6: Sequence[int]) -> Tuple[Optional[int], List[int], bool]:
    top_votes = get_top_special_votes(conn, issue_no, top_n=8)
    if not top_votes:
        return None, [], False
    mains = {int(n) for n in main6}
    recent_3_specials = [int(r["special_number"]) for r in conn.execute(
        "SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 3"
    ).fetchall()]
    primary = None
    for n in top_votes:
        n_int = int(n)
        if n_int not in mains and n_int not in recent_3_specials:
            primary = n_int
            break
    if primary is None:
        primary = int(top_votes[0])
    conflict = primary in mains
    defenses = []
    for n in top_votes:
        n_int = int(n)
        if n_int == primary or n_int in defenses:
            continue
        if n_int in mains:
            continue
        if n_int in recent_3_specials:
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
        "SELECT special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 30"
    ).fetchall()]
    omission = {n: 31 for n in ALL_NUMBERS}
    for idx, n in enumerate(recent_specials):
        omission[n] = min(omission.get(n, 31), idx + 1)

    mains = {int(x) for x in main6}
    candidate_scores: Dict[int, float] = {}
    for n in sorted(set(specials)):
        zodiac = get_zodiac_by_number(n)
        score = 0.0
        score += number_votes.get(n, 0) * 2.2
        score += weighted_scores.get(n, 0.0) * 2.0
        score += zodiac_counter.get(zodiac, 0) * 0.9
        score += min(1.2, float(omission.get(n, 31)) / 25.0)
        if n in mains:
            score -= 1.2
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
        return specials, zodiac_list, None, None
    return specials, zodiac_list, best, get_zodiac_by_number(best)


def _weighted_consensus_pools(conn: sqlite3.Connection, issue_no: str) -> Tuple[List[int], List[int], List[int], List[int], Optional[int]]:
    strategy_weights = get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT)
    number_scores: Dict[int, float] = {}
    special_scores: Dict[int, float] = {}

    for strategy in STRATEGY_IDS:
        run = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (issue_no, strategy),
        ).fetchone()
        if not run:
            continue
        run_id = int(run["id"])
        w = float(strategy_weights.get(strategy, 1.0 / len(STRATEGY_IDS)))
        pool20 = get_pool_numbers_for_run(conn, run_id, 20)
        for idx, n in enumerate(pool20):
            if not (1 <= int(n) <= 49):
                continue
            rank_boost = (20 - idx) / 20.0
            number_scores[int(n)] = number_scores.get(int(n), 0.0) + w * rank_boost

        main6 = get_pool_numbers_for_run(conn, run_id, 6)
        for n in main6:
            if 1 <= int(n) <= 49:
                number_scores[int(n)] = number_scores.get(int(n), 0.0) + w * 0.35

        _, special = get_picks_for_run(conn, run_id)
        if special is not None and 1 <= int(special) <= 49:
            special_scores[int(special)] = special_scores.get(int(special), 0.0) + w

    if not number_scores:
        return [], [], [], [], None

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


# ========== 三中三入口 ==========
def get_trio_from_merged_pool20(conn: sqlite3.Connection, issue_no: str) -> List[int]:
    return get_trio_from_merged_pool20_v2(conn, issue_no)


# ========== 最终推荐函数 ==========
def get_final_recommendation(conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT issue_no FROM prediction_runs WHERE status='PENDING' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    issue_no = row["issue_no"]

    main6, pool10, pool14, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if not main6 or not pool10 or not pool14 or not pool20:
        return None
    special, special_defenses, special_conflict = get_special_recommendation(conn, issue_no, main6)
    if special is None:
        return None
    strategy_specials, strategy_special_zodiacs, strategy_strong_special, strategy_strong_zodiac = get_strong_special_from_strategies(
        conn, issue_no, main6
    )

    predict_trio = get_trio_from_merged_pool20(conn, issue_no)

    zodiac_single = get_single_zodiac_pick(conn, issue_no, window=16)
    zodiac_two = get_two_zodiac_picks(conn, issue_no, window=16)
    return (
        issue_no,
        main6,
        special,
        pool10,
        pool14,
        pool20,
        predict_trio,
        special_defenses,
        special_conflict,
        zodiac_single,
        zodiac_two,
        strategy_specials,
        strategy_special_zodiacs,
        strategy_strong_special,
        strategy_strong_zodiac,
    )


def print_final_recommendation(conn: sqlite3.Connection) -> None:
    rec = get_final_recommendation(conn)
    if not rec:
        print("\n最终推荐: (暂无有效预测)")
        return
    issue_no, main6, special, pool10, pool14, pool20, predict_trio, special_defenses, special_conflict, zodiac_single, zodiac_two, strategy_specials, strategy_special_zodiacs, strategy_strong_special, strategy_strong_zodiac = rec
    special_text = _fmt_num(special)
    p6 = " ".join(_fmt_num(n) for n in main6)
    p10 = " ".join(_fmt_num(n) for n in pool10)
    p14 = " ".join(_fmt_num(n) for n in pool14)
    p20 = " ".join(_fmt_num(n) for n in pool20)
    trio_str = " ".join(_fmt_num(n) for n in predict_trio) if predict_trio else "无"

    zodiac_single_text = zodiac_single if zodiac_single else "数据不足"
    zodiac_two_text = "、".join(zodiac_two) if zodiac_two else "数据不足"
    defense_text = " ".join(_fmt_num(n) for n in special_defenses) if special_defenses else "无"
    strategy_special_text = " ".join(_fmt_num(n) for n in strategy_specials) if strategy_specials else "无"
    strategy_zodiac_text = "、".join(strategy_special_zodiacs) if strategy_special_zodiacs else "无"
    strong_special_text = _fmt_num(strategy_strong_special) if strategy_strong_special is not None else "无"
    strong_zodiac_text = strategy_strong_zodiac if strategy_strong_zodiac else "无"

    print("\n" + "=" * 50)
    print(f"【最终推荐 - 期号 {issue_no}】")
    print(f"策略说明: 主号采用「多策略加权共识」(基于最近{FEATURE_WINDOW_DEFAULT}期特征 + 近{WEIGHT_WINDOW_DEFAULT}期动态权重)，特别号采用「加权投票」")
    print(f"  6号池 : {p6} | 特别号: {special_text}")
    print(f"  10号池: {p10} | 特别号: {special_text}")
    print(f"  14号池: {p14} | 特别号: {special_text}")
    print(f"  20号池: {p20} | 特别号: {special_text}")
    print(f"特别号建议: 主推 {special_text} | 防守 {defense_text}")
    print(f"六策略特别号组: {strategy_special_text}")
    print(f"六策略生肖组: {strategy_zodiac_text}")
    print(f"六策略极强号: {strong_special_text} ({strong_zodiac_text})")
    if special_conflict:
        print("特别号提示: 主推候选与主号冲突，已自动切换到非冲突号码")
    print(f"三中三预测（综合20码池+动态权重）: {trio_str}")
    print(f"[zodiac] 2生肖推荐: {zodiac_two_text}")
    print(f"[zodiac] 1生肖推荐: {zodiac_single_text}")
    print("=" * 50)


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
    lines.append(f"[stats] 复盘最新一期 {issue_no}（{draw_date}）")
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


def print_dashboard(conn: sqlite3.Connection) -> None:
    latest = get_latest_draw(conn)
    if latest:
        nums = " ".join(_fmt_num(n) for n in json.loads(latest["numbers_json"]))
        print(f"最新开奖: {latest['issue_no']} {latest['draw_date']} | 主号: {nums} | 特别号: {_fmt_num(int(latest['special_number']))}")
    else:
        print("暂无开奖数据。")

    print_recommendation_sheet(conn, limit=8)

    print("\n策略平均命中率:")
    stats = get_review_stats(conn)
    if not stats:
        print("  (暂无复盘)")
    for s in stats:
        strategy_name = STRATEGY_LABELS.get(s["strategy"], s["strategy"])
        print(
            f"  - {strategy_name}: 次数={s['c']} 平均命中={s['avg_hit']:.2f} "
            f"命中率6={s['avg_rate'] * 100:.2f}% 10={float(s['avg_rate_10'] or 0) * 100:.2f}% "
            f"14={float(s['avg_rate_14'] or 0) * 100:.2f}% 20={float(s['avg_rate_20'] or 0) * 100:.2f}% "
            f"特别号命中率={s['special_rate'] * 100:.2f}% 至少中1个={s['hit1_rate'] * 100:.2f}% 至少中2个={s['hit2_rate'] * 100:.2f}%"
        )

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

    zodiac_report = get_recent_single_zodiac_report(conn, lookback=20, history_window=16)
    print("\n单生肖复盘（最近20期）:")
    print(
        f"  - 最近样本={int(zodiac_report['samples'])}期 "
        f"命中率={zodiac_report['hit_rate'] * 100:.1f}% "
        f"最大连空={int(zodiac_report['max_miss_streak'])}"
    )
    zodiac_two_report = get_recent_two_zodiac_report(conn, lookback=20, history_window=16)
    print("双生肖复盘（最近20期）:")
    print(
        f"  - 最近样本={int(zodiac_two_report['samples'])}期 "
        f"命中率={zodiac_two_report['hit_rate'] * 100:.1f}% "
        f"最大连空={int(zodiac_two_report['max_miss_streak'])}"
    )

    print_final_recommendation(conn)

    print("\n" + review_latest_prediction(conn))

    if PUSHPLUS_TOKEN:
        rec = get_final_recommendation(conn)
        if rec:
            issue_no, main6, special, _, _, _, predict_trio, special_defenses, special_conflict, zodiac_single, zodiac_two, strategy_specials, strategy_special_zodiacs, strategy_strong_special, strategy_strong_zodiac = rec
            special_text = _fmt_num(special)
            trio_str = " ".join(_fmt_num(n) for n in predict_trio) if predict_trio else "无"
            defense_text = " ".join(_fmt_num(n) for n in special_defenses) if special_defenses else "无"
            strong_special_text = _fmt_num(strategy_strong_special) if strategy_strong_special is not None else "无"
            strong_zodiac_text = strategy_strong_zodiac if strategy_strong_zodiac else "无"
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

            zodiac_single_text = zodiac_single if zodiac_single else "数据不足"
            zodiac_two_text = "、".join(zodiac_two) if zodiac_two else "数据不足"
            conflict_tip = "（已避开主号冲突）" if special_conflict else ""

            content = (
                f"【新澳门·{issue_no}期推荐】\n"
                f"2生肖推荐：{zodiac_two_text}\n"
                f"1生肖推荐：{zodiac_single_text}\n"
                f"特别号主推：{special_text}{conflict_tip}\n"
                f"特别号防守：{defense_text}\n"
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
        records = fetch_macau_records(timeout=args.api_timeout, retries=args.api_retries)
        if args.require_continuity:
            missing = missing_issues_since_latest(conn, records)
            if missing:
                raise RuntimeError(
                    f"Continuity check failed. Missing {len(missing)} issues, sample={','.join(missing[:10])}"
                )
        total, inserted, updated = sync_from_records(conn, records, source="macau_api")
        mined_cfg = ensure_mined_pattern_config(conn, force=args.remine)
        reviewed = review_latest(conn)
        bt_issues, bt_runs = 0, 0
        if args.with_backtest:
            bt_issues, bt_runs = run_historical_backtest(conn, rebuild=False, max_issues=BACKTEST_ISSUES_DEFAULT)
        issue = generate_predictions(conn)
        patched = backfill_missing_special_picks(conn)
        print(f"Sync done. total={total}, inserted={inserted}, updated={updated}, reviewed={reviewed}, next_prediction={issue}")
        print(f"Mined config: {json.dumps(mined_cfg, ensure_ascii=False)}")
        if bt_issues > 0:
            print(f"Backtest updated. issues={bt_issues}, strategy_runs={bt_runs}")
        if patched > 0:
            print(f"Patched missing special picks: {patched}")
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
        print_dashboard(conn)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="新澳门六合彩预测工具 - v4全面优化版")
    p.add_argument("--db", default=DB_PATH_DEFAULT, help=f"SQLite db path (default: {DB_PATH_DEFAULT})")
    p.add_argument("--update", action="store_true", help="Quick sync from API (same as sync)")
    p.add_argument("--remine", action="store_true", help="Re-mine pattern config before sync/backtest")
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

    p_predict = sub.add_parser("predict", help="Generate predictions for next or specified issue")
    p_predict.add_argument("--issue", help="Target issue, e.g. 26/023")
    p_predict.set_defaults(func=cmd_predict)

    p_review = sub.add_parser("review", help="Review pending runs for latest or specified issue")
    p_review.add_argument("--issue", help="Issue to review, e.g. 26/022")
    p_review.set_defaults(func=cmd_review)

    p_show = sub.add_parser("show", help="Show local dashboard summary")
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

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.update:
        cmd_sync(args)
        return
    if not args.command:
        parser.error("Please provide a subcommand, or use --update.")
    args.func(args)


if __name__ == "__main__":
    main()
