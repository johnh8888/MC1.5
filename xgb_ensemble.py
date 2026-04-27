import json
import pickle
from collections import Counter

import numpy as np
import xgboost as xgb

XGB_MODEL_PATH = "xgb_ensemble_model.pkl"

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}


def get_zodiac_by_number(num):
    for z, nums in ZODIAC_MAP.items():
        if num in nums:
            return z
    return "马"


def _calc_omission(num, recent):
    for i, d in enumerate(recent):
        if num in d:
            return i + 1
    return len(recent) + 1


def _zodiac_hot(num, counter):
    if not counter:
        return 0.0
    return counter.get(get_zodiac_by_number(num), 0) / sum(counter.values())


def _neighbor_dist(num, all_nums):
    if not all_nums:
        return 49
    return min(abs(num - n) for n in all_nums)


def _tail_match(num, last_sp):
    return 1 if last_sp and num % 10 == last_sp % 10 else 0


def build_features(scores_dict, num, temp, recent, last_sp, zod_cnt):
    f = []
    # 原有5个旧策略评分 → 全部删除，不再使用
    # f.append(...)  这五行注释掉

    # 保留号码数值和温度
    f.append(num / 49.0)
    f.append(temp.get("cold_ratio", 0.0))
    f.append(temp.get("zone_entropy", 0.0))

    # 保留4个核心新特征
    f.append(_calc_omission(num, recent[-10:]) / max(1, len(recent[-10:])))
    f.append(_zodiac_hot(num, zod_cnt))
    all_recent = [n for d in recent[-10:] for n in d]
    f.append(_neighbor_dist(num, all_recent) / 49.0)
    f.append(_tail_match(num, last_sp))

    # 保留2个对抗特征
    last_draw = recent[-1] if recent else []
    last_odd = sum(1 for n in last_draw if n % 2 == 1)
    odd_bias = abs(last_odd - 3) / 3.0
    f.append(odd_bias)

    is_neighbor = 0
    if last_draw:
        for n in last_draw:
            if num in (n - 1, n + 1):
                is_neighbor = 1
                break
    f.append(is_neighbor)

    return np.array(f, dtype=np.float32)


def compute_market_temperature(draws):
    if len(draws) < 5:
        return {"cold_ratio": 0.0, "zone_entropy": 0.0}
    recent = draws[:20]
    omission = {}
    for i, d in enumerate(recent):
        for n in d:
            if n not in omission:
                omission[n] = i + 1
    cold = sum(1 for v in omission.values() if v >= 8) / 49.0
    zc = [0] * 5
    for d in recent[:10]:
        for n in d:
            zc[min(4, (n - 1) // 10)] += 1
    total = sum(zc) or 1
    probs = [c / total for c in zc if c > 0]
    ent = -sum(p * np.log(p) for p in probs) if probs else 0.0
    return {"cold_ratio": cold, "zone_entropy": ent}


def train_ensemble_model(conn, gen_fn):
    """训练模型，仅需 conn 和策略生成函数 gen_fn"""
    rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date").fetchall()
    X, Y = [], []
    for i in range(10, len(rows)):
        true_set = set(json.loads(rows[i][0]))
        recent = [json.loads(r[0]) for r in rows[i - 10:i]]
        scores = {}
        for s in ["hot_v1", "cold_rebound_v1", "momentum_v1", "balanced_v1", "pattern_mined_v1"]:
            _, _, _, sm = gen_fn(recent, s, conn=conn, issue_no="backtest")
            scores[s] = sm
        temp = compute_market_temperature(recent)
        last_sp = int(rows[i - 1][1]) if i > 0 else None
        all_nums = [n for d in recent for n in d] + [int(r[1]) for r in rows[i - 10:i]]
        zod_cnt = Counter()
        for n in all_nums:
            zod_cnt[get_zodiac_by_number(n)] += 1
        for num in range(1, 50):
            X.append(build_features(scores, num, temp, recent, last_sp, zod_cnt))
            Y.append(1 if num in true_set else 0)
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        scale_pos_weight=6,
        colsample_bytree=0.7,
        subsample=0.7,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(np.array(X), np.array(Y))
    importances = model.feature_importances_
    print("[XGB] 特征重要性:", importances)
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("[XGB] 增强特征模型已训练并保存")
    return model


def load_or_train_ensemble_model(conn, gen_fn):
    """加载已有模型，若无则调用训练，不再需要 load_fn"""
    try:
        with open(XGB_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return train_ensemble_model(conn, gen_fn)


def ensemble_predict(scores_dict, temp, model, recent, last_sp, zod_cnt):
    probs = {}
    for num in range(1, 50):
        f = build_features(scores_dict, num, temp, recent, last_sp, zod_cnt)
        probs[num] = model.predict_proba(f.reshape(1, -1))[0][1]
    return probs
