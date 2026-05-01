#!/usr/bin/env python3
"""HMM潜伏状态特征 —— 捕捉生肖序列的冷热转换"""
import sqlite3, json, pickle
import numpy as np
from pathlib import Path
from collections import Counter

ZODIAC_LIST = ["马","蛇","龙","兔","虎","牛","鼠","猪","狗","鸡","猴","羊"]
ZODIAC_MAP = {z: i for i, z in enumerate(ZODIAC_LIST)}

def get_zodiac_by_number(n):
    for z, nums in {
        "马": [1,13,25,37,49],"蛇":[2,14,26,38],"龙":[3,15,27,39],
        "兔":[4,16,28,40],"虎":[5,17,29,41],"牛":[6,18,30,42],
        "鼠":[7,19,31,43],"猪":[8,20,32,44],"狗":[9,21,33,45],
        "鸡":[10,22,34,46],"猴":[11,23,35,47],"羊":[12,24,36,48]
    }.items():
        if n in nums:
            return z
    return "马"

def connect_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def build_zodiac_sequence(conn):
    rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date ASC").fetchall()
    seq = []
    for row in rows:
        nums = json.loads(row["numbers_json"])
        sp = int(row["special_number"])
        zodiacs = [get_zodiac_by_number(n) for n in nums] + [get_zodiac_by_number(sp)]
        cnt = Counter(zodiacs)
        dominant = cnt.most_common(1)[0][0]
        seq.append(ZODIAC_MAP[dominant])
    return np.array(seq)

def train_hmm(conn, model_path='hmm_zodiac.pkl', n_states=3):
    try:
        from hmmlearn import hmm
    except ImportError:
        print("[HMM] hmmlearn 未安装，跳过训练")
        return None
    observations = build_zodiac_sequence(conn)
    if len(observations) < 50:
        print("[HMM] 数据不足，跳过训练")
        return None
    obs = observations.reshape(-1, 1)
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=200, tol=0.001, verbose=False)
    model.fit(obs)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[HMM] 模型已保存至 {model_path}")
    return model

def get_hmm_state_proba(conn, model_path='hmm_zodiac.pkl'):
    """返回下一期各生肖的出现概率（基于HMM状态后验与发射矩阵）"""
    if not Path(model_path).exists():
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    observations = build_zodiac_sequence(conn)
    if len(observations) < 2:
        return None
    try:
        seq_len = min(15, len(observations))
        recent = observations[-seq_len:].reshape(-1, 1)
        state_proba = model.predict_proba(recent)[-1]  # 最后一步的状态概率
        emission = model.emissionprob_                 # (n_states, 12)
        zodiac_proba = np.dot(state_proba, emission)
        zodiac_proba /= zodiac_proba.sum()
        return dict(zip(ZODIAC_LIST, zodiac_proba))
    except Exception as e:
        print(f"[HMM] 预测失败: {e}")
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--n_states', type=int, default=3)
    args = parser.parse_args()
    conn = connect_db(args.db)
    train_hmm(conn, n_states=args.n_states)
    conn.close()
