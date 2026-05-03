#!/usr/bin/env python3
"""三生肖严格预测模块 - 供优化器使用"""
import json
from collections import Counter
from pathlib import Path
from typing import List

ZODIAC_MAP = {
    "马": [1,13,25,37,49], "蛇": [2,14,26,38], "龙": [3,15,27,39],
    "兔": [4,16,28,40], "虎": [5,17,29,41], "牛": [6,18,30,42],
    "鼠": [7,19,31,43], "猪": [8,20,32,44], "狗": [9,21,33,45],
    "鸡": [10,22,34,46], "猴": [11,23,35,47], "羊": [12,24,36,48],
}
def get_zodiac_by_number(n: int) -> str:
    for z, nums in ZODIAC_MAP.items():
        if n in nums:
            return z
    return "马"

def get_three_zodiac_picks(conn, lookback: int = 30, **kwargs) -> List[str]:
    params_path = Path(__file__).parent / "best_params_zodiac.json"
    params = {}
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    lstm_weight = float(params.get("three_lstm_weight", 0.3))
    hmm_weight = float(params.get("three_hmm_weight", 0.2))
    lstm_seq_len = int(params.get("lstm_seq_len", 30)) if params.get("three_lstm_seq_len") is None else int(params["three_lstm_seq_len"])

    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT ?",
        (lookback,)
    ).fetchall()
    if not rows:
        return ["马", "蛇", "龙"]

    scores = {z: 0.0 for z in ZODIAC_MAP}
    for idx, row in enumerate(rows):
        recency = 1.0 / (1.0 + idx * 0.1)
        nums = json.loads(row["numbers_json"])
        for n in nums:
            scores[get_zodiac_by_number(n)] += 0.8 * recency
        sp = row["special_number"]
        scores[get_zodiac_by_number(sp)] += 2.2 * recency

    # 可选 LSTM 增强
    try:
        from lstm_predictor import predict_lstm_proba
        lstm_probs = predict_lstm_proba(conn, seq_len=lstm_seq_len)
        if lstm_probs:
            for z in scores:
                scores[z] = (1 - lstm_weight) * scores[z] + lstm_weight * lstm_probs.get(z, 0.0)
    except Exception:
        pass

    # 可选 HMM 增强
    try:
        from hmm_features import get_hmm_state_proba
        hmm_probs = get_hmm_state_proba(conn)
        if hmm_probs:
            for z in scores:
                scores[z] = (1 - hmm_weight) * scores[z] + hmm_weight * hmm_probs.get(z, 0.0)
    except Exception:
        pass

    # 遗漏值奖励
    omission = {z: len(rows)+1 for z in ZODIAC_MAP}
    for i, row in enumerate(rows):
        nums = json.loads(row["numbers_json"])
        sp = row["special_number"]
        appeared = {get_zodiac_by_number(n) for n in nums}
        appeared.add(get_zodiac_by_number(sp))
        for z in appeared:
            if omission[z] > i+1:
                omission[z] = i+1
    for z, om in omission.items():
        if om >= 6:
            scores[z] += min(2.5, om / 4.0)

    sorted_z = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [z for z, _ in sorted_z[:3]]
