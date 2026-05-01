#!/usr/bin/env python3
"""断点续传式自我进化优化器 —— 目标：一生肖≥90% 二生肖≥92% 特别生肖≥90% 最大连空≤1"""
import sqlite3, json, sys, argparse, time, os
from collections import Counter
import optuna

# 完整的生肖映射
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
    conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row; return conn

def load_issues(conn, recent=300):
    rows = conn.execute("SELECT issue_no,draw_date,numbers_json,special_number FROM draws ORDER BY draw_date ASC").fetchall()
    return [(r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])) for r in rows[-recent:]]

def pred_single(hist, wsize, rec_w, safe_th):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = hist[-wsize:] if len(hist) >= wsize else hist
    for idx, (_, nums, sp) in enumerate(recent[::-1]):
        w = rec_w / (1.0 + idx * 0.15)
        for n in nums: scores[get_zodiac(n)] += w
        scores[get_zodiac(sp)] += w * 2.0
    if max(scores.values()) < safe_th:
        omission = {z: 0 for z in ZODIAC_MAP}
        for i in range(len(recent)):
            _, nums, sp = recent[-(i+1)]
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
        _, nums, sp = hist[-(i+1)]
        for z in ZODIAC_MAP: omission[z] = omission.get(z, i+1) if omission[z]==0 else omission[z]
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
    for z in omission: omission[z] *= four_boost
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

def evaluate(issues, params):
    single_h = two_h = four_h = 0
    single_streak = two_streak = four_streak = 0
    max_single_streak = max_two_streak = max_four_streak = 0
    total = 0
    for i in range(60, len(issues)):
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums)
        cur_zod.add(get_zodiac(cur_sp))

        s = pred_single(past, params['wsize'], params['rec_w'], params['safe_th'])
        if s in cur_zod: single_h += 1; single_streak = 0
        else: single_streak += 1; max_single_streak = max(max_single_streak, single_streak)

        two = pred_two(past)
        if any(z in cur_zod for z in two): two_h += 1; two_streak = 0
        else: two_streak += 1; max_two_streak = max(max_two_streak, two_streak)

        four = pred_four(past, params['four_boost'])
        if any(z in cur_zod for z in four): four_h += 1; four_streak = 0
        else: four_streak += 1; max_four_streak = max(max_four_streak, four_streak)

        total += 1

    if total == 0: return 0.0, 0,0,0,0,0,0
    r1 = single_h / total
    r2 = two_h / total
    r4 = four_h / total
    max_streak = max(max_single_streak, max_two_streak, max_four_streak)

    # 硬指标
    if r1 < 0.90 or r2 < 0.92 or r4 < 0.90 or max_streak > 1:
        return 0.0, r1, r2, r4, max_single_streak, max_two_streak, max_four_streak

    avg_hit = (r1 + r2 + r4) / 3
    return avg_hit + 1.0/(1.0+max_streak), r1, r2, r4, max_single_streak, max_two_streak, max_four_streak

def objective(trial, issues):
    p = {
        'wsize': trial.suggest_int('wsize', 4, 15),
        'rec_w': trial.suggest_float('rec_w', 0.3, 2.5),
        'safe_th': trial.suggest_float('safe_th', 0.8, 2.0),
        'four_boost': trial.suggest_float('four_boost', 0.5, 5.0),
    }
    score, _, _, _, _, _, _ = evaluate(issues, p)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--trials', type=int, default=200)
    args = parser.parse_args()

    conn = connect_db(args.db)
    issues = load_issues(conn, recent=300)
    conn.close()
    if len(issues) < 80:
        print("数据不足，至少需要80期历史数据，请先同步。")
        sys.exit(1)

    # 断点续传存储
    study = optuna.create_study(
        direction='maximize',
        study_name='ultimate_optimizer',
        storage='sqlite:///optuna_study.db',
        load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, issues), n_trials=args.trials, show_progress_bar=True)

    best_p = study.best_params
    score, r1, r2, r4, ms1, ms2, ms4 = evaluate(issues, best_p)
    print(f"当前最佳: 一生肖={r1:.3f}(连空{ms1}) 二肖={r2:.3f}(连空{ms2}) 四肖={r4:.3f}(连空{ms4})")

    with open("best_params_zodiac.json", "w") as f:
        json.dump(best_p, f, indent=2)
    if score > 0:
        print("🎉 已达标！")
    else:
        print("未达标，但已保存当前最佳参数。下次运行将从此继续搜索。")

if __name__ == "__main__":
    main()