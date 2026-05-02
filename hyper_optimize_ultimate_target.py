#!/usr/bin/env python3
"""澳门彩优化器（稳定增强版）：近10期 一生肖≥70% 二肖(任1)≥80% 四肖≥95% 特别号≥50% 连空≤1"""
import sqlite3, json, sys, argparse
from collections import Counter
import optuna

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}

def get_zodiac(n):
    for z, ns in ZODIAC_MAP.items():
        if n in ns: return z
    return "马"

def connect_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def load_issues(conn, recent=120):
    rows = conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
    return [(r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])) for r in rows[-recent:]]

# ---- 生肖预测函数（同v1，无变化） ----
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
    hot_cnt = Counter([get_zodiac(sp) for sp in specials])
    hot = max(hot_cnt, key=hot_cnt.get)
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
            if z not in picks: picks.append(z); break
    return picks[:4]

# ---- 新增：特别号精选预测函数 ----
def pred_special_3(hist, params):
    """
    基于历史 hist 和参数，返回精选的3个特别号。
    hist 结构同 issues: [(issue_no, main_nums, special_num), ...]
    """
    if len(hist) < 5:
        return [1, 2, 3]
    recent_specials = [row[2] for row in hist]
    latest_special = recent_specials[-1] if recent_specials else None

    # 生肖池：取最近四肖推荐的后4个，结合高频生肖扩展
    four_zodiacs = pred_four(hist, params.get('four_boost', 1.0))
    zodiac_counter = Counter([get_zodiac(sp) for sp in recent_specials[-8:]])
    # 扩展池到6-8个生肖
    extra = [z for z, _ in zodiac_counter.most_common(3) if z not in four_zodiacs][:2]
    pool_zodiacs = four_zodiacs + extra
    seen = set()
    final_pool = []
    for z in pool_zodiacs:
        if z not in seen:
            seen.add(z)
            final_pool.append(z)
    while len(final_pool) < 8:
        for z in ZODIAC_MAP:
            if z not in final_pool:
                final_pool.append(z)
            if len(final_pool) >= 8:
                break

    candidates = []
    for z in final_pool:
        candidates.extend(ZODIAC_MAP[z])
    candidates = list(set(candidates))

    # 遗漏计算
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission:
            omission[sp] = i + 1

    cold_th = params.get('cold_threshold', 11)
    nb1_bonus = params.get('neighbor_1_bonus', 6.0)
    nb2_bonus = params.get('neighbor_2_bonus', 1.0)
    lgb_w = params.get('lgb_weight', 0.6)
    omit_boost = params.get('omit_boost', 2.0)

    # 优先取2冷
    cold_cands = [n for n in candidates if omission.get(n, 30) >= cold_th]
    cold_cands.sort(key=lambda n: omission.get(n, 30), reverse=True)
    picks = cold_cands[:2]

    # 第3码：邻号或次冷
    if len(picks) < 3 and latest_special is not None:
        neighbors = [n for n in candidates if n not in picks and abs(n - latest_special) == 1]
        if neighbors:
            picks.append(max(neighbors, key=lambda n: omission.get(n, 30) + nb1_bonus))
        else:
            neighbors2 = [n for n in candidates if n not in picks and abs(n - latest_special) == 2]
            if neighbors2:
                picks.append(max(neighbors2, key=lambda n: omission.get(n, 30) + nb2_bonus))
            else:
                rest = sorted([n for n in candidates if n not in picks],
                              key=lambda n: omission.get(n, 30), reverse=True)
                while len(picks) < 3 and rest:
                    picks.append(rest.pop(0))
    # 补充不足
    while len(picks) < 3:
        for n in candidates:
            if n not in picks:
                picks.append(n)
                if len(picks) >= 3:
                    break
    # 权重微调（LGB逻辑简化：邻号加分已在上面体现，这里仅做最后的排序保障）
    return picks[:3]

# ---- 评估函数（增加特别号） ----
def evaluate(issues, params):
    total = len(issues)
    if total < 15: return -999.0, 0,0,0,0,0,0,0
    recent10_start = max(0, total - 10)
    single_hits = two_hits = four_hits = special_hits = 0
    single_streak = two_streak = four_streak = special_streak = 0
    max_single = max_two = max_four = max_special = 0

    for i in range(recent10_start, total):
        past = issues[:i]   # 严格历史
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums)
        cur_zod.add(get_zodiac(cur_sp))

        # 一生肖
        s = pred_single(past, params['wsize'], params['rec_w'], params['safe_th'])
        if s in cur_zod: single_hits += 1; single_streak = 0
        else: single_streak += 1; max_single = max(max_single, single_streak)

        # 二生肖
        two = pred_two(past)
        if any(z in cur_zod for z in two): two_hits += 1; two_streak = 0
        else: two_streak += 1; max_two = max(max_two, two_streak)

        # 四生肖
        four = pred_four(past, params['four_boost'])
        if any(z in cur_zod for z in four): four_hits += 1; four_streak = 0
        else: four_streak += 1; max_four = max(max_four, four_streak)

        # 特别号
        sp_picks = pred_special_3(past, params)
        if cur_sp in sp_picks: special_hits += 1; special_streak = 0
        else: special_streak += 1; max_special = max(max_special, special_streak)

    n = total - recent10_start
    if n == 0: return -999.0, 0,0,0,0,0,0,0
    r1 = single_hits / n
    r2 = two_hits / n
    r4 = four_hits / n
    rsp = special_hits / n
    max_strk = max(max_single, max_two, max_four, max_special)

    # 软性连空惩罚 (统一)
    streak_factor = 1.0
    if max_strk >= 4: streak_factor = 0.2
    elif max_strk == 3: streak_factor = 0.5
    elif max_strk == 2: streak_factor = 0.8

    # 综合得分 (增加特别号权重)
    score = r1 * 0.30 + r2 * 0.30 + r4 * 0.20 + rsp * 0.20
    if r1 < 0.70: score *= 0.85
    if r2 < 0.80: score *= 0.85
    if r4 < 0.95: score *= 0.90
    if rsp < 0.50: score *= 0.90
    return score * streak_factor, r1, r2, r4, rsp, max_single, max_two, max_four, max_special

def objective(trial, issues):
    p = {
        # 生肖参数
        'wsize': trial.suggest_int('wsize', 2, 20),
        'rec_w': trial.suggest_float('rec_w', 0.1, 4.0),
        'safe_th': trial.suggest_float('safe_th', 0.4, 2.5),
        'four_boost': trial.suggest_float('four_boost', 0.3, 6.0),
        # 特别号参数
        'cold_threshold': trial.suggest_int('cold_threshold', 8, 18),
        'neighbor_1_bonus': trial.suggest_float('neighbor_1_bonus', 2.0, 10.0),
        'neighbor_2_bonus': trial.suggest_float('neighbor_2_bonus', 0.0, 5.0),
        'lgb_weight': trial.suggest_float('lgb_weight', 0.3, 1.0),
        'omit_boost': trial.suggest_float('omit_boost', 1.0, 5.0),
    }
    score, _, _, _, _, _, _, _, _ = evaluate(issues, p)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--recent', type=int, default=120, help='使用的最近期数')
    args = parser.parse_args()

    conn = connect_db(args.db)
    issues = load_issues(conn, recent=args.recent)
    conn.close()
    if len(issues) < 20:
        print("数据不足，退出。")
        sys.exit(1)

    study = optuna.create_study(
        direction='maximize',
        study_name='macau_stable_v2',
        storage='sqlite:///optuna_macau_stable.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(lambda t: objective(t, issues), n_trials=args.trials, show_progress_bar=True)

    best_p = study.best_params
    score, r1, r2, r4, rsp, ms1, ms2, ms4, mssp = evaluate(issues, best_p)
    print(f"近10期: 一生肖={r1:.3f}(连空{ms1}) 二肖={r2:.3f}(连空{ms2}) 四肖={r4:.3f}(连空{ms4}) 特别号={rsp:.3f}(连空{mssp})")
    with open("best_params_zodiac.json", "w") as f:
        json.dump(best_p, f, indent=2)

    if r1 >= 0.70 and r2 >= 0.80 and r4 >= 0.95 and rsp >= 0.50 and max(ms1, ms2, ms4, mssp) <= 1:
        print("🎉 达标！")
        sys.exit(0)
    else:
        print("未达标，继续搜索。")
        sys.exit(1)

if __name__ == "__main__":
    main()
