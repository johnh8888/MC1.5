#!/usr/bin/env python3
"""澳门彩优化器（修正版）：特别号预测匹配主脚本逻辑，目标近10期 一生肖≥70% 二肖≥80% 四肖≥95% 特别号≥50% 连空≤1"""
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

# ---- 生肖预测函数（保持不变） ----
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
    for i, (_, nums, sp) in enumerate(hist[::-1]):
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

# ---- 特别号精选预测（与主脚本 get_precise_specials_from_history 完全一致） ----
def pred_special_from_history(history, zodiac_pool, params, top_n=3):
    if not zodiac_pool:
        return []
    latest_row = history[0]
    latest_special = latest_row[2]  # 特别号
    recent_specials = [r[2] for r in history[:12]]

    omission = {}
    for i, sp in enumerate(recent_specials):
        if sp not in omission:
            omission[sp] = i + 1

    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z, [])))
    if not candidates:
        return []

    # 尾数统计
    tail_counter = Counter()
    for row in history[:8]:
        for n in row[1]:            # 主号
            tail_counter[n % 10] += 1
    for sp in recent_specials[:8]:
        tail_counter[sp % 10] += 3

    hot_tails = {t for t, _ in tail_counter.most_common(6)}
    last_tail = latest_special % 10
    neighbor_tails = {last_tail, (last_tail + 1) % 10, (last_tail - 1) % 10}

    selected = []
    penalty_nums = set(recent_specials[:2])

    # 邻号1优先
    neighbors = [n for n in candidates if abs(n - latest_special) == 1 and n not in penalty_nums]
    if not neighbors:
        neighbors = [n for n in candidates if abs(n - latest_special) == 1]
    if neighbors:
        selected.append(max(neighbors, key=lambda n: omission.get(n, 20)))

    # 尾数匹配
    if len(selected) < top_n:
        tail_candidates = [n for n in candidates if n not in selected and n % 10 == last_tail and n not in penalty_nums]
        if not tail_candidates:
            tail_candidates = [n for n in candidates if n not in selected and n % 10 in neighbor_tails and n not in penalty_nums]
        if not tail_candidates:
            tail_candidates = [n for n in candidates if n not in selected and n % 10 in hot_tails and n not in penalty_nums]
        if not tail_candidates:
            tail_candidates = [n for n in candidates if n not in selected and n % 10 == last_tail]
        if tail_candidates:
            selected.append(max(tail_candidates, key=lambda n: omission.get(n, 20)))

    # 邻号2
    if len(selected) < top_n:
        neighbors2 = [n for n in candidates if abs(n - latest_special) == 2 and n not in selected and n not in penalty_nums]
        if not neighbors2:
            neighbors2 = [n for n in candidates if abs(n - latest_special) == 2 and n not in selected]
        if neighbors2:
            selected.append(max(neighbors2, key=lambda n: omission.get(n, 20)))

    # 若还不够，选遗漏最大值
    if len(selected) < top_n:
        cold_pool = [n for n in candidates if n not in selected and n != latest_special]
        if cold_pool:
            selected.append(max(cold_pool, key=lambda n: omission.get(n, 20)))

    # 仍不足则填充
    if len(selected) < top_n:
        remaining = [n for n in candidates if n not in selected]
        remaining.sort(key=lambda n: omission.get(n, 20), reverse=True)
        for n in remaining:
            selected.append(n)
            if len(selected) >= top_n:
                break

    # 应用优化参数微调（仅对已选号码的排序作补充，但不改变已选集合）
    # 注：原主脚本中此处会使用 cold_threshold, neighbor_1_bonus 等，但 get_precise_specials_from_history 本身固定规则。
    # 为保持与主脚本一致，此处直接返回 selected[:top_n]
    return selected[:top_n]

def build_zodiac_pool(hist):
    """生成生肖池，与主脚本 backfill_special_picks_log 中的构建逻辑一致"""
    base_four = pred_four(hist, 1.0)  # 使用默认1.0，后文会乘 four_boost，这里只取基础四肖
    recent_zodiacs = [get_zodiac(r[2]) for r in hist[:8]]
    zodiac_freq = Counter(recent_zodiacs)
    specials_hist = [r[2] for r in hist[:30]]
    omission_zodiac = {z: 0 for z in ZODIAC_MAP}
    for idx, sp in enumerate(specials_hist):
        z = get_zodiac(sp)
        if omission_zodiac[z] == 0:
            omission_zodiac[z] = idx + 1
    sorted_omit = sorted(omission_zodiac.items(), key=lambda x: -x[1])
    extra_freq = [z for z, _ in zodiac_freq.most_common(3) if z not in base_four][:2]
    extra_cold = [z for z, _ in sorted_omit if z not in base_four and z not in extra_freq][:2]
    last3_zodiacs = [get_zodiac(r[2]) for r in hist[:3]]
    latest_main = hist[0][1]
    main_counter = Counter(get_zodiac(n) for n in latest_main)
    top_main = main_counter.most_common(1)[0][0] if main_counter else None
    zodiac_pool = base_four + extra_freq + extra_cold + last3_zodiacs + ([top_main] if top_main else [])
    seen = set()
    final_pool = []
    for z in zodiac_pool:
        if z not in seen:
            seen.add(z)
            final_pool.append(z)
    while len(final_pool) < 8:
        for z in ZODIAC_MAP:
            if z not in final_pool:
                final_pool.append(z)
            if len(final_pool) >= 8:
                break
    return final_pool[:8]

# ---- 评估函数 ----
def evaluate(issues, params):
    total = len(issues)
    if total < 15:
        return -999.0, 0,0,0,0,0,0,0
    recent10_start = max(0, total - 10)
    single_hits = two_hits = four_hits = special_hits = 0
    single_streak = two_streak = four_streak = special_streak = 0
    max_single = max_two = max_four = max_special = 0

    for i in range(recent10_start, total):
        past = issues[:i]
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

        # 特别号：匹配主脚本的真实逻辑
        zodiac_pool = build_zodiac_pool(past)
        sp_picks = pred_special_from_history(past, zodiac_pool, params, top_n=3)
        if cur_sp in sp_picks: special_hits += 1; special_streak = 0
        else: special_streak += 1; max_special = max(max_special, special_streak)

    n = total - recent10_start
    if n == 0:
        return -999.0, 0,0,0,0,0,0,0
    r1 = single_hits / n
    r2 = two_hits / n
    r4 = four_hits / n
    rsp = special_hits / n
    max_strk = max(max_single, max_two, max_four, max_special)

    streak_factor = 1.0
    if max_strk >= 4: streak_factor = 0.2
    elif max_strk == 3: streak_factor = 0.5
    elif max_strk == 2: streak_factor = 0.8

    score = r1 * 0.30 + r2 * 0.30 + r4 * 0.20 + rsp * 0.20
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
        # 特别号参数（尽管当前使用的主逻辑是固定规则，但保留以供将来扩展）
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
