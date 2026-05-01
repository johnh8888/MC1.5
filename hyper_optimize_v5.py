#!/usr/bin/env python3
""" 终极生肖超参优化器 v5 —— 目标：一生肖≥85%、二生肖≥95%、特别生肖≥85% """
import sqlite3, json, sys, argparse
from collections import Counter
import optuna
import numpy as np

# 正确的生肖字典（不是 set）
ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}
ALL_NUMS = list(range(1, 50))

def get_zodiac(n):
    for z, ns in ZODIAC_MAP.items():
        if n in ns:
            return z
    return "马"

def connect_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def load_issues(conn, recent=200):
    rows = conn.execute("SELECT issue_no,draw_date,numbers_json,special_number FROM draws ORDER BY draw_date ASC").fetchall()
    return [(r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])) for r in rows[-recent:]]

# ---------- 预测函数 ----------
def pred_single(hist, wsize, rec_w, safe_th):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = hist[-wsize:] if len(hist) >= wsize else hist
    for idx, (_, nums, sp) in enumerate(recent[::-1]):
        w = rec_w / (1.0 + idx * 0.15)
        for n in nums:
            scores[get_zodiac(n)] += w
        scores[get_zodiac(sp)] += w * 2.0
    if max(scores.values()) < safe_th:
        omission = {z: 0 for z in ZODIAC_MAP}
        for i in range(len(recent)):
            _, nums, sp = recent[-(i + 1)]
            for z in ZODIAC_MAP:
                if omission[z] == 0:
                    omission[z] = i + 1
            for n in nums:
                omission[get_zodiac(n)] = 0
            omission[get_zodiac(sp)] = 0
        return max(omission.items(), key=lambda x: x[1])[0]
    return max(scores.items(), key=lambda x: x[1])[0]

def pred_two(hist):
    specials = [sp for _, _, sp in hist[-10:]]
    hot_counter = Counter([get_zodiac(sp) for sp in specials])
    hot = max(hot_counter, key=hot_counter.get)
    omission = {z: 0 for z in ZODIAC_MAP}
    for i in range(len(hist)):
        _, nums, sp = hist[-(i + 1)]
        for z in ZODIAC_MAP:
            if omission[z] == 0:
                omission[z] = i + 1
        for n in nums:
            omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    cold = max((z for z in ZODIAC_MAP if z != hot), key=lambda z: omission[z])
    return [hot, cold]

def pred_four(hist, four_boost):
    omission = {z: 0 for z in ZODIAC_MAP}
    specials = [sp for _, _, sp in hist]
    for i, sp in enumerate(specials[::-1]):
        z = get_zodiac(sp)
        if omission[z] == 0:
            omission[z] = i + 1
    for z in omission:
        omission[z] *= four_boost
    sorted_cold = sorted(omission.items(), key=lambda x: (-x[1], x[0]))
    picks = [z for z, _ in sorted_cold[:3]]
    latest_z = get_zodiac(specials[-1]) if specials else None
    if latest_z and latest_z not in picks:
        picks.append(latest_z)
    else:
        for z, _ in sorted_cold[3:]:
            if z not in picks:
                picks.append(z)
                break
    return picks[:4]

# ---------- 评估函数（强力惩罚不达标） ----------
def evaluate(issues, params):
    single_h = two_h = four_h = 0
    total = 0
    min_len = 60
    for i in range(min_len, len(issues)):
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums)
        cur_zod.add(get_zodiac(cur_sp))

        s = pred_single(past, params['wsize'], params['rec_w'], params['safe_th'])
        if s in cur_zod:
            single_h += 1

        two = pred_two(past)
        if any(z in cur_zod for z in two):
            two_h += 1

        four = pred_four(past, params['four_boost'])
        if any(z in cur_zod for z in four):
            four_h += 1

        total += 1

    if total == 0:
        return 0.0
    r1 = single_h / total
    r2 = two_h / total
    r4 = four_h / total
    score = (r1 + r2 + r4) / 3
    # 强力惩罚
    if r1 < 0.85:
        score *= 0.6
    if r2 < 0.95:
        score *= 0.7
    if r4 < 0.85:
        score *= 0.6
    return score

def objective(trial, issues):
    p = {
        'wsize': trial.suggest_int('wsize', 4, 15),
        'rec_w': trial.suggest_float('rec_w', 0.3, 2.5),
        'safe_th': trial.suggest_float('safe_th', 0.8, 2.0),
        'four_boost': trial.suggest_float('four_boost', 0.5, 5.0),
    }
    return evaluate(issues, p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--trials', type=int, default=500)
    args = parser.parse_args()
    conn = connect_db(args.db)
    issues = load_issues(conn, recent=200)
    conn.close()
    if len(issues) < 80:
        print("数据不足（至少需要80期），请先同步数据。")
        sys.exit(1)
    study = optuna.create_study(
        direction='maximize',
        study_name='zodiac_v5',
        storage='sqlite:///optuna_v5.db',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50)
    )
    study.optimize(lambda t: objective(t, issues), n_trials=args.trials, show_progress_bar=True)
    best = study.best_params
    best['score'] = study.best_value
    print("最佳生肖参数(目标90%命中率):")
    for k, v in best.items():
        print(f"  {k}: {v}")
    with open("best_params_zodiac.json", "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
