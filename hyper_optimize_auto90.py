#!/usr/bin/env python3
"""
智能自适应优化器 —— 目标：一生肖≥85%、二生肖≥95%、特别生肖≥85%
自动反复调参，直到达标或尝试次数耗尽。
"""

import sqlite3, json, sys, argparse, time
from collections import Counter
import optuna
import numpy as np

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}
ALL_NUMS = list(range(1, 50))

def get_zodiac(n):
    for z, ns in ZODIAC_MAP.items():
        if n in ns: return z
    return "马"

def connect_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def load_issues(conn, recent=200):
    rows = conn.execute("SELECT issue_no,draw_date,numbers_json,special_number FROM draws ORDER BY draw_date ASC").fetchall()
    return [(r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])) for r in rows[-recent:]]

# ---------- 预测函数 (带可变参数) ----------
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
                if omission[z] == 0: omission[z] = i + 1
            for n in nums: omission[get_zodiac(n)] = 0
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
            if omission[z] == 0: omission[z] = i + 1
        for n in nums: omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    cold = max((z for z in ZODIAC_MAP if z != hot), key=lambda z: omission[z])
    return [hot, cold]

def pred_four(hist, four_boost):
    omission = {z: 0 for z in ZODIAC_MAP}
    specials = [sp for _, _, sp in hist]
    for i, sp in enumerate(specials[::-1]):
        z = get_zodiac(sp)
        if omission[z] == 0: omission[z] = i + 1
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

# ---------- 评估函数（惩罚系数动态可调） ----------
def evaluate(issues, params, penalty_factors):
    single_h = two_h = four_h = 0
    total = 0
    min_len = 60
    for i in range(min_len, len(issues)):
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums)
        cur_zod.add(get_zodiac(cur_sp))

        s = pred_single(past, params['wsize'], params['rec_w'], params['safe_th'])
        if s in cur_zod: single_h += 1

        two = pred_two(past)
        if any(z in cur_zod for z in two): two_h += 1

        four = pred_four(past, params['four_boost'])
        if any(z in cur_zod for z in four): four_h += 1

        total += 1

    if total == 0: return 0.0, 0, 0, 0
    r1 = single_h / total
    r2 = two_h / total
    r4 = four_h / total
    # 得分 = 加权平均，权重固定为各1/3，但用惩罚因子大幅拉低不达标者
    score = (r1 + r2 + r4) / 3
    # 强制惩罚：低于目标则得分乘以惩罚系数（越小越严厉）
    if r1 < 0.85: score *= penalty_factors[0]
    if r2 < 0.95: score *= penalty_factors[1]
    if r4 < 0.85: score *= penalty_factors[2]
    return score, r1, r2, r4

def objective(trial, issues, penalty_factors):
    p = {
        'wsize': trial.suggest_int('wsize', 4, 15),
        'rec_w': trial.suggest_float('rec_w', 0.3, 2.5),
        'safe_th': trial.suggest_float('safe_th', 0.8, 2.0),
        'four_boost': trial.suggest_float('four_boost', 0.5, 5.0),
    }
    score, _, _, _ = evaluate(issues, p, penalty_factors)
    return score

def run_optimization(issues, penalty_factors, n_trials, study_name_prefix):
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{study_name_prefix}_{int(time.time())}',
        storage='sqlite:///optuna_auto90.db',
        load_if_exists=False,  # 每次全新搜索
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
    )
    study.optimize(lambda t: objective(t, issues, penalty_factors), n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    # 用最佳参数评估真实命中率
    score, r1, r2, r4 = evaluate(issues, best_params, penalty_factors)
    return best_params, score, r1, r2, r4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--max_rounds', type=int, default=5, help='最大自动调整轮数')
    parser.add_argument('--trials_per_round', type=int, default=200, help='每轮试验次数')
    args = parser.parse_args()

    conn = connect_db(args.db)
    issues = load_issues(conn, recent=200)
    conn.close()
    if len(issues) < 80:
        print("数据不足（至少需要80期），请先同步数据。")
        sys.exit(1)

    # 初始惩罚因子（越小越严厉，未达标会乘以此因子）
    penalty_factors = [0.7, 0.7, 0.7]  # 对应一生肖、二生肖、特别生肖
    best_overall_params = None
    best_overall_score = -1
    best_r1 = best_r2 = best_r4 = 0

    for round_idx in range(1, args.max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"第 {round_idx} 轮自动调整，当前惩罚系数：单={penalty_factors[0]}, 二={penalty_factors[1]}, 四={penalty_factors[2]}")
        print(f"{'='*60}")

        params, score, r1, r2, r4 = run_optimization(
            issues, penalty_factors, args.trials_per_round, f"round{round_idx}"
        )
        print(f"\n本轮结果：一生肖={r1:.3f}  二生肖={r2:.3f}  特别生肖={r4:.3f}  得分={score:.4f}")
        print(f"最佳参数：{params}")

        # 保存历史最佳
        if score > best_overall_score:
            best_overall_score = score
            best_overall_params = params
            best_r1, best_r2, best_r4 = r1, r2, r4

        # 判断是否达标 (一生肖≥0.85、二生肖≥0.95、特别生肖≥0.85)
        if r1 >= 0.85 and r2 >= 0.95 and r4 >= 0.85:
            print(f"🎉 命中率已全部达标！停止调整。")
            break

        # 未达标：加强惩罚（降低未达标者的系数）
        new_penalty = penalty_factors.copy()
        if r1 < 0.85:
            new_penalty[0] = max(0.2, penalty_factors[0] - 0.15)  # 更严厉
        if r2 < 0.95:
            new_penalty[1] = max(0.2, penalty_factors[1] - 0.15)
        if r4 < 0.85:
            new_penalty[2] = max(0.2, penalty_factors[2] - 0.15)
        penalty_factors = new_penalty

    # 输出最终结果
    print("\n" + "="*60)
    print("最终最佳参数 (基于最高得分):")
    for k, v in best_overall_params.items():
        print(f"  {k}: {v}")
    print(f"真实命中率: 一生肖={best_r1:.3f}({best_r1*100:.1f}%)  二生肖={best_r2:.3f}({best_r2*100:.1f}%)  特别生肖={best_r4:.3f}({best_r4*100:.1f}%)")
    print(f"最佳得分: {best_overall_score:.4f}")
    with open("best_params_zodiac.json", "w") as f:
        json.dump(best_overall_params, f, indent=2)
    print("参数已保存到 best_params_zodiac.json")

if __name__ == "__main__":
    main()
