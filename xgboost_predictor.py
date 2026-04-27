import json
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

ALL_NUMBERS = list(range(1, 50))
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


def get_zodiac_by_number(n: int) -> str:
    for z, nums in ZODIAC_MAP.items():
        if n in nums:
            return z
    return "马"


class XGBoostPredictor:
    def __init__(self):
        self.model = None

    def _build_features(self, conn, train_up_to_issue: Optional[str] = None):
        draws = conn.execute(
            "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date, issue_no"
        ).fetchall()
        if train_up_to_issue:
            draws = [d for d in draws if d['issue_no'] <= train_up_to_issue]

        rows = []
        for idx in range(12, len(draws)):
            current = draws[idx]
            history = draws[idx - 12:idx]
            winning_nums = set(json.loads(current['numbers_json']))

            freq_6 = Counter()
            freq_12 = Counter()
            for h in history[-6:]:
                for n in json.loads(h['numbers_json']):
                    freq_6[n] += 1
            for h in history:
                for n in json.loads(h['numbers_json']):
                    freq_12[n] += 1

            omission = {}
            for n in ALL_NUMBERS:
                omit = 0
                for h_idx in range(idx - 1, -1, -1):
                    h = draws[h_idx]
                    if n in json.loads(h['numbers_json']):
                        break
                    omit += 1
                omission[n] = omit

            last_nums = json.loads(draws[idx - 1]['numbers_json'])
            last_special = draws[idx - 1]['special_number']
            last_set = set(last_nums + [last_special])
            neighbor_1 = set()
            neighbor_2 = set()
            for b in last_set:
                for d in (-2, -1, 1, 2):
                    nb = b + d
                    if 1 <= nb <= 49:
                        if abs(d) == 1:
                            neighbor_1.add(nb)
                        else:
                            neighbor_2.add(nb)

            for num in ALL_NUMBERS:
                rows.append({
                    'num': num,
                    'freq_6': freq_6.get(num, 0),
                    'freq_12': freq_12.get(num, 0),
                    'omission': omission.get(num, 0),
                    'is_neighbor_1': 1 if num in neighbor_1 else 0,
                    'is_neighbor_2': 1 if num in neighbor_2 else 0,
                    'zodiac_code': list(ZODIAC_MAP.keys()).index(get_zodiac_by_number(num)),
                    'tail': num % 10,
                    'parity': num % 2,
                    'zone': (num - 1) // 10,
                    'target': 1 if num in winning_nums else 0,
                })

        return pd.DataFrame(rows)

    def train(self, conn):
        df = self._build_features(conn)
        feature_cols = [c for c in df.columns if c not in ('num', 'target')]
        X = df[feature_cols]
        y = df['target']
        scale_pos_weight = len(y[y == 0]) / max(1, len(y[y == 1]))
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
        self.model.fit(X, y)
        return self

    def predict_pool(self, conn, top_k=20):
        if not self.model:
            raise RuntimeError("Model not trained yet. Run train() first.")

        draws = conn.execute(
            "SELECT issue_no, numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 13"
        ).fetchall()
        if len(draws) < 13:
            raise RuntimeError("Need at least 13 draws for features.")

        history = draws[1:13]
        freq_6 = Counter()
        freq_12 = Counter()
        for h in history[-6:]:
            for n in json.loads(h['numbers_json']):
                freq_6[n] += 1
        for h in history:
            for n in json.loads(h['numbers_json']):
                freq_12[n] += 1

        omission = {}
        all_prev_draws = conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date DESC").fetchall()
        for n in ALL_NUMBERS:
            omit = 0
            for d in all_prev_draws[1:]:
                if n in json.loads(d['numbers_json']):
                    break
                omit += 1
            omission[n] = omit

        last_nums = json.loads(history[-1]['numbers_json'])
        last_special = history[-1]['special_number']
        last_set = set(last_nums + [last_special])
        neighbor_1 = set()
        neighbor_2 = set()
        for b in last_set:
            for d in (-2, -1, 1, 2):
                nb = b + d
                if 1 <= nb <= 49:
                    if abs(d) == 1:
                        neighbor_1.add(nb)
                    else:
                        neighbor_2.add(nb)

        rows = []
        for num in ALL_NUMBERS:
            rows.append({
                'freq_6': freq_6.get(num, 0),
                'freq_12': freq_12.get(num, 0),
                'omission': omission.get(num, 0),
                'is_neighbor_1': 1 if num in neighbor_1 else 0,
                'is_neighbor_2': 1 if num in neighbor_2 else 0,
                'zodiac_code': list(ZODIAC_MAP.keys()).index(get_zodiac_by_number(num)),
                'tail': num % 10,
                'parity': num % 2,
                'zone': (num - 1) // 10,
            })
        X_latest = pd.DataFrame(rows)
        probs = self.model.predict_proba(X_latest)[:, 1]
        numbers_probs = list(zip(ALL_NUMBERS, probs))
        numbers_probs.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in numbers_probs[:top_k]]
