import json
from collections import Counter
from typing import Optional

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency on some runners
    pd = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency on some runners
    xgb = None

# ======== 策略得分计算所需公用函数（从主脚本提取） ========
ALL_NUMBERS = list(range(1, 50))

STRATEGY_WEIGHT_CONFIGS = {
    "hot_v1":          {"window": 6,  "w_freq": 0.74, "w_omit": 0.06, "w_mom": 0.14, "w_zone": 0.06, "w_adj": 0.10},
    "cold_rebound_v1": {"window": 13, "w_freq": 0.06, "w_omit": 0.62, "w_mom": 0.22, "w_zone": 0.05, "w_adj": 0.12},
    "momentum_v1":     {"window": 7,  "w_freq": 0.10, "w_omit": 0.05, "w_mom": 0.75, "w_zone": 0.05, "w_adj": 0.05},
    "balanced_v1":     {"window": 10, "w_freq": 0.36, "w_omit": 0.26, "w_mom": 0.18, "w_zone": 0.06, "w_adj": 0.14},
    "pattern_mined_v1": {"window": 6,  "w_freq": 0.30, "w_omit": 0.45, "w_mom": 0.15, "w_zone": 0.10, "w_adj": 0.10},
}


def get_zodiac_by_number(n: int) -> str:
    zodiac_map = {
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
    for z, nums in zodiac_map.items():
        if n in nums:
            return z
    return "马"


def _normalize_scores(score_map):
    values = list(score_map.values())
    mn, mx = min(values), max(values)
    if mx == mn:
        return {k: 0.0 for k in score_map}
    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def _freq_map(draws):
    freq = {n: 0.0 for n in ALL_NUMBERS}
    for draw in draws:
        for n in draw:
            freq[n] += 1.0
    return freq


def _omission_map(draws):
    omission = {n: float(len(draws) + 1) for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        for n in draw:
            omission[n] = min(omission[n], float(i + 1))
    return omission


def _momentum_map(draws):
    m = {n: 0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        w = 1.0 / (1.0 + i)
        for n in draw:
            m[n] += w
    return m


def _zone_heat_map(draws, window=3):
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


def _adjacency_compensation_map(draws, window=5):
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


def compute_strategy_scores(history_draws, strategy_name):
    """返回策略对每个号码的原始得分（未归一化）"""
    cfg = STRATEGY_WEIGHT_CONFIGS.get(strategy_name)
    if not cfg:
        return {n: 0.0 for n in ALL_NUMBERS}
    window_size = int(cfg.get("window", 6))
    window = history_draws[:max(3, window_size)]
    freq = _normalize_scores(_freq_map(window))
    omission = _normalize_scores(_omission_map(window))
    momentum = _normalize_scores(_momentum_map(window))
    zone = _normalize_scores(_zone_heat_map(window, window=min(3, len(window))))
    adjacency = _normalize_scores(_adjacency_compensation_map(window, window=min(5, len(window))))

    w_freq = cfg.get("w_freq", 0.40)
    w_omit = cfg.get("w_omit", 0.28)
    w_mom = cfg.get("w_mom", 0.16)
    w_zone = cfg.get("w_zone", 0.06)
    w_adj = cfg.get("w_adj", 0.10)

    scores = {}
    for n in ALL_NUMBERS:
        scores[n] = (
            freq[n] * w_freq
            + omission[n] * w_omit
            + momentum[n] * w_mom
            + zone[n] * w_zone
            + adjacency[n] * w_adj
        )
    return scores


class XGBoostPredictor:
    def __init__(self):
        self.model = None

    def _build_features(self, conn, train_up_to_issue=None):
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

            # 原有的频率/遗漏/邻号计算保持
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

            # ----- 新增：计算五个策略的得分 -----
            history_draws = [json.loads(h['numbers_json']) for h in history]
            strategy_scores = {}
            non_ensemble = ["hot_v1", "cold_rebound_v1", "momentum_v1", "balanced_v1", "pattern_mined_v1"]
            for sname in non_ensemble:
                ss = compute_strategy_scores(history_draws, sname)
                strategy_scores[sname] = ss

            for num in ALL_NUMBERS:
                features = {
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
                }
                for sname in non_ensemble:
                    features[f'score_{sname}'] = strategy_scores[sname].get(num, 0.0)
                features['target'] = 1 if num in winning_nums else 0
                rows.append(features)

        df = pd.DataFrame(rows)
        return df

    def train(self, conn):
        df = self._build_features(conn)
        feature_cols = [c for c in df.columns if c not in ('num', 'target')]
        X = df[feature_cols]
        y = df['target']
        scale_pos_weight = len(y[y == 0]) / max(1, len(y[y == 1]))
        if xgb is None:
            raise RuntimeError("xgboost is required for XGBoost training")
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

        # 获取最近13期记录（前12期作为历史）
        draws = conn.execute(
            "SELECT issue_no, numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 13"
        ).fetchall()
        if len(draws) < 13:
            raise RuntimeError("Need at least 13 draws for features.")

        # 历史 = 最近12期（不包含最新一期）
        history_draws_data = [json.loads(d['numbers_json']) for d in draws[1:13]]

        # 最新一期开奖的号码，用于 neighbor 计算
        latest_nums = json.loads(draws[0]['numbers_json'])
        latest_special = draws[0]['special_number']
        latest_set = set(latest_nums + [latest_special])

        # 计算频率、遗漏等特征（用所有历史数据）
        all_prev = conn.execute(
            "SELECT numbers_json FROM draws ORDER BY draw_date DESC"
        ).fetchall()
        freq_6 = Counter()
        freq_12 = Counter()
        for h in draws[1:7]:
            for n in json.loads(h['numbers_json']):
                freq_6[n] += 1
        for h in draws[1:13]:
            for n in json.loads(h['numbers_json']):
                freq_12[n] += 1

        omission = {}
        for n in ALL_NUMBERS:
            omit = 0
            for d in all_prev[1:]:
                if n in json.loads(d['numbers_json']):
                    break
                omit += 1
            omission[n] = omit

        neighbor_1 = set()
        neighbor_2 = set()
        for b in latest_set:
            for d in (-2, -1, 1, 2):
                nb = b + d
                if 1 <= nb <= 49:
                    if abs(d) == 1:
                        neighbor_1.add(nb)
                    else:
                        neighbor_2.add(nb)

        # 计算策略得分（基于最近12期历史）
        non_ensemble = ["hot_v1", "cold_rebound_v1", "momentum_v1", "balanced_v1", "pattern_mined_v1"]
        strategy_scores = {}
        for sname in non_ensemble:
            strategy_scores[sname] = compute_strategy_scores(history_draws_data, sname)

        # 构建特征向量
        rows = []
        for num in ALL_NUMBERS:
            feature = {
                'freq_6': freq_6.get(num, 0),
                'freq_12': freq_12.get(num, 0),
                'omission': omission.get(num, 0),
                'is_neighbor_1': 1 if num in neighbor_1 else 0,
                'is_neighbor_2': 1 if num in neighbor_2 else 0,
                'zodiac_code': list(ZODIAC_MAP.keys()).index(get_zodiac_by_number(num)),
                'tail': num % 10,
                'parity': num % 2,
                'zone': (num - 1) // 10,
            }
            for sname in non_ensemble:
                feature[f'score_{sname}'] = strategy_scores[sname].get(num, 0.0)
            rows.append(feature)

        X_latest = pd.DataFrame(rows)
        if hasattr(self.model, 'feature_names_in_'):
            X_latest = X_latest[self.model.feature_names_in_]
        probs = self.model.predict_proba(X_latest)[:, 1]
        numbers_probs = list(zip(ALL_NUMBERS, probs))
        numbers_probs.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in numbers_probs[:top_k]]
