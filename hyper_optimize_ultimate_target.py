#!/usr/bin/env python3
"""
澳门彩优化器（最终版）
完全匹配主脚本的特别号预测逻辑，支持调试打印
"""
import sqlite3, json, sys, argparse
from collections import Counter
import optuna

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}

DEBUG = False   # 由 --debug 参数设置

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

# ------ 生肖预测（同上） ------
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
    if not hot_cnt:
        return ["马", "蛇"]
    hot = max(hot_cnt, key=hot_cnt.get)
    omission = {z: 0 for z in ZODIAC_MAP}
    for i, (_, nums, sp) in enumerate(hist[::-1]):
        for z in ZODIAC_MAP: omission[z] = omission.get(z, i+1) if omission[z]==0 else omission[z]
        for n in nums: omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    # 找出另一个遗漏最大的（不同于hot）
    others = [(z, omission[z]) for z in ZODIAC_MAP if z != hot]
    cold = max(others, key=lambda x: x[1])[0]
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

# ------ 特别号预测（复制自 get_precise_specials_for_issue，参数驱动） ------
def pred_special_online(hist, zodiac_pool, params, top_n=3):
    if not zodiac_pool:
        return []
    recent_specials = [row[2] for row in hist[::-1]]   # 从近到远
    if not recent_specials:
        return list(set(ZODIAC_MAP.get(zodiac_pool[0], [])))[:top_n]
    latest_special = recent_specials[0]
    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission:
            omission[sp] = i + 1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    if not candidates:
        return []
    cold_threshold = int(params.get('cold_threshold', 11))
    neighbor_1_bonus = float(params.get('neighbor_1_bonus', 6.918))
    neighbor_2_bonus = float(params.get('neighbor_2_bonus', 0.514))
    penalty_coeff = float(params.get('penalty_coeff', 0.76))
    lgb_weight = float(params.get('lgb_weight', 0.6146))
    omit_boost = float(params.get('four_omit_boost', 2.578))
    # 2冷
    cold_picks = sorted(
        [n for n in candidates if omission.get(n, 20) >= cold_threshold],
        key=lambda n: omission.get(n, 20), reverse=True
    )[:2]
    while len(cold_picks) < 2:
        remaining = [n for n in candidates if n not in cold_picks]
        if not remaining: break
        cold_picks.append(max(remaining, key=lambda n: omission.get(n, 20)))
    picks = cold_picks[:2]
    # 补足到 top_n
    if len(picks) < top_n:
        neighbors = [n for n in candidates if abs(n - latest_special) == 1 and n not in picks]
        if neighbors:
            picks.append(max(neighbors, key=lambda n: omission.get(n, 20) + neighbor_1_bonus))
    if len(picks) < top_n:
        neighbors2 = [n for n in candidates if abs(n - latest_special) == 2 and n not in picks]
        if neighbors2:
            picks.append(max(neighbors2, key=lambda n: omission.get(n, 20) + neighbor_2_bonus))
    while len(picks) < top_n:
        rest = sorted([n for n in candidates if n not in picks],
                      key=lambda n: omission.get(n, 20), reverse=True)
        if rest: picks.append(rest[0])
        else: break
    # 加权排序
    if recent_specials:
        scored = []
        for n in picks:
            score = float(omission.get(n, 20)) * lgb_weight
            if n in recent_specials[:3]:
                score *= penalty_coeff
            if omission.get(n, 20) >= cold_threshold:
                score += omit_boost
            scored.append((n, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        picks = [n for n, _ in scored]
    return picks[:top_n]

def build_zodiac_pool(hist, params):
    """模拟线上 enhanced_zodiacs 的生肖池"""
    base_four = pred_four(hist, params['four_boost'])
    specials_hist = [r[2] for r in hist]
    recent_zodiacs = [get_zodiac(sp) for sp in specials_hist[-8:]]
    zodiac_freq = Counter(recent_zodiacs)
    extra_freq = [z for z, _ in zodiac_freq.most_common(3) if z not in base_four][:2]
    omission_z = {z: 0 for z in ZODIAC_MAP}
    for idx, sp in enumerate(specials_hist[::-1]):
        z = get_zodiac(sp)
        if omission_z[z] == 0: omission_z[z] = idx + 1
    sorted_omit = sorted(omission_z.items(), key=lambda x: -x[1])
    extra_cold = [z for z, _ in sorted_omit if z not in base_four and z not in extra_freq][:2]
    last3 = [get_zodiac(r[2]) for r in hist[-3:]]
    union = base_four + extra_freq + extra_cold + last3
    seen = set()
    pool = []
    for z in union:
        if z not in seen:
            seen.add(z)
            pool.append(z)
    for z in ZODIAC_MAP:
        if len(pool) >= 8: break
        if z not in pool: pool.append(z)
    return pool[:8]

# ------ 评估函数 ------
def evaluate(issues, params):
    total = len(issues)
    if total < 15: return -999.0, 0,0,0,0,0,0,0
    recent10_start = max(0, total - 10)
    single_hits = two_hits = four_hits = special_hits = 0
    single_streak = two_streak = four_streak = special_streak = 0
    max_single = max_two = max_four = max_special = 0
    for i in range(recent10_start, total):
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums) | {get_zodiac(cur_sp)}
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
        zodiac_pool = build_zodiac_pool(past, params)
        sp_picks = pred_special_online(past, zodiac_pool, params, top_n=3)
        if cur_sp in sp_picks: special_hits += 1; special_streak = 0
        else: special_streak += 1; max_special = max(max_special, special_streak)
        if DEBUG and i >= recent10_start:
            print(f"[DEBUG] 期号 {issues[i][0]} 历史期数 {len(past)} 生肖池 {zodiac_pool} 预测特号 {sp_picks} 实际 {cur_sp}")
    n = total - recent10_start
    if n == 0: return -999.0, 0,0,0,0,0,0,0
    r1 = single_hits / n
    r2 = two_hits / n
    r4 = four_hits / n
    rsp = special_hits / n
    max_strk = max(max_single, max_two, max_four, max_special)
    streak_factor = 1.0
    if max_strk >= 4: streak_factor = 0.2
    elif max_strk == 3: streak_factor = 0.5
    elif max_strk == 2: streak_factor = 0.8
    score = r1*0.30 + r2*0.30 + r4*0.20 + rsp*0.20
    if r1 < 0.70: score *= 0.85
    if r2 < 0.80: score *= 0.85
    if r4 < 0.95: score *= 0.90
    if rsp < 0.50: score *= 0.90
    return score * streak_factor, r1, r2, r4, rsp, max_single, max_two, max_four, max_special

def objective(trial, issues):
    p = {
        'wsize': trial.suggest_int('wsize', 2, 20),
        'rec_w': trial.suggest_float('rec_w', 0.1, 4.0),
        'safe_th': trial.suggest_float('safe_th', 0.4, 2.5),
        'four_boost': trial.suggest_float('four_boost', 0.3, 6.0),
        'cold_threshold': trial.suggest_int('cold_threshold', 8, 18),
        'neighbor_1_bonus': trial.suggest_float('neighbor_1_bonus', 2.0, 10.0),
        'neighbor_2_bonus': trial.suggest_float('neighbor_2_bonus', 0.0, 5.0),
        'penalty_coeff': trial.suggest_float('penalty_coeff', 0.5, 1.0),
        'lgb_weight': trial.suggest_float('lgb_weight', 0.3, 1.0),
        'four_omit_boost': trial.suggest_float('four_omit_boost', 1.0, 5.0),
    }
    score, _, _, _, _, _, _, _, _ = evaluate(issues, p)
    return score

def main():
    global DEBUG
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--recent', type=int, default=120)
    parser.add_argument('--debug', action='store_true', help='打印每期调试信息')
    args = parser.parse_args()
    DEBUG = args.debug

    conn = connect_db(args.db)
    issues = load_issues(conn, recent=args.recent)
    conn.close()
    if len(issues) < 20:
        print("数据不足，退出."); sys.exit(1)
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
        print("🎉 达标！"); sys.exit(0)
    else:
        print("未达标，继续搜索."); sys.exit(1)

if __name__ == "__main__":
    main()
