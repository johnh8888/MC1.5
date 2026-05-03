#!/usr/bin/env python3
"""
香港六合彩全自动策略进化优化器
一生肖6策略、二生肖6策略、三生肖5策略、特肖(四只)6策略
独立评估，互不干扰
"""

import sqlite3, json, sys, argparse, random
from collections import Counter
import optuna

ZODIAC_MAP = {
    "马": [1,13,25,37,49], "蛇": [2,14,26,38], "龙": [3,15,27,39],
    "兔": [4,16,28,40], "虎": [5,17,29,41], "牛": [6,18,30,42],
    "鼠": [7,19,31,43], "猪": [8,20,32,44], "狗": [9,21,33,45],
    "鸡": [10,22,34,46], "猴": [11,23,35,47], "羊": [12,24,36,48],
}
ZODIAC_PAIR = {
    "鼠":"牛","牛":"鼠","虎":"猪","猪":"虎","兔":"狗","狗":"兔",
    "龙":"鸡","鸡":"龙","蛇":"猴","猴":"蛇","马":"羊","羊":"马"
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
        "SELECT issue_no,draw_date,numbers_json,special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
    return [(r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])) for r in rows[-recent:]]

# ---------- 基础工具函数 ----------
def _zodiac_omission_map(rows):
    zod_omis = {z: len(rows)+1 for z in ZODIAC_MAP}
    for i, (_, nums, sp) in enumerate(rows):
        appeared = {get_zodiac(n) for n in nums} | {get_zodiac(sp)}
        for z in appeared:
            if zod_omis[z] > i+1:
                zod_omis[z] = i+1
    return zod_omis

def _build_zodiac_scores_from_rows(rows, decay=0.08):
    scores = {z:0.0 for z in ZODIAC_MAP}
    omission = _zodiac_omission_map(rows)
    for idx, (_, nums, sp) in enumerate(rows):
        w = 1.0/(1.0+idx*decay)
        for n in nums: scores[get_zodiac(n)] += w
        scores[get_zodiac(sp)] += 1.8*w
    for z in scores:
        omit = omission.get(z, len(rows))
        if omit >= 8: scores[z] += 4.0
        elif omit >= 3: scores[z] += omit/6.0
    return scores

# ---------- 一生肖策略 (6) ----------
def single_weighted(hist, wsize, rec_w, safe_th):
    scores = {z:0.0 for z in ZODIAC_MAP}
    recent = hist[-wsize:] if len(hist)>=wsize else hist
    for idx,(_,nums,sp) in enumerate(recent[::-1]):
        w = rec_w/(1.0+idx*0.15)
        for n in nums: scores[get_zodiac(n)] += w
        scores[get_zodiac(sp)] += w*2.0
    if max(scores.values())<safe_th:
        omission = _zodiac_omission_map(recent)
        return max(omission.items(),key=lambda x:x[1])[0]
    return max(scores.items(),key=lambda x:x[1])[0]

def single_pure_hot(hist, wsize, rec_w, safe_th):
    scores = {z:0.0 for z in ZODIAC_MAP}
    recent = hist[-wsize:] if len(hist)>=wsize else hist
    for idx,(_,nums,sp) in enumerate(recent[::-1]):
        w = rec_w/(1.0+idx*0.1)
        for n in nums: scores[get_zodiac(n)] += w
    return max(scores.items(),key=lambda x:x[1])[0]

def single_pure_cold(hist, wsize, rec_w, safe_th):
    omission = _zodiac_omission_map(hist)
    return max(omission.items(),key=lambda x:x[1])[0]

def single_hybrid(hist, wsize, rec_w, safe_th):
    hot = single_pure_hot(hist, wsize, rec_w, safe_th)
    cold = single_pure_cold(hist, wsize, rec_w, safe_th)
    return hot if random.random()<0.6 else cold

def single_hot_main_only(hist, wsize, rec_w, safe_th):
    scores = {z:0.0 for z in ZODIAC_MAP}
    recent = hist[-wsize:] if len(hist)>=wsize else hist
    for idx,(_,nums,sp) in enumerate(recent[::-1]):
        w = rec_w/(1.0+idx*0.1)
        for n in nums: scores[get_zodiac(n)] += w
    return max(scores.items(),key=lambda x:x[1])[0]

def single_last_special(hist, wsize, rec_w, safe_th):
    if hist: return get_zodiac(hist[-1][2])
    return "马"

STRATEGIES_SINGLE = {
    "weighted": single_weighted,
    "pure_hot": single_pure_hot,
    "pure_cold": single_pure_cold,
    "hybrid": single_hybrid,
    "hot_main_only": single_hot_main_only,
    "last_special": single_last_special,
}

# ---------- 二生肖策略 (6) ----------
def two_hot_cold(hist):
    specials = [sp for _,_,sp in hist[-10:]]
    hot_cnt = Counter([get_zodiac(sp) for sp in specials])
    hot = max(hot_cnt, key=hot_cnt.get)
    omission = _zodiac_omission_map(hist)
    cold = max((z for z in ZODIAC_MAP if z!=hot), key=lambda z: omission[z])
    return [hot, cold]

def two_double_hot(hist):
    specials = [sp for _,_,sp in hist[-8:]]
    hot_cnt = Counter([get_zodiac(sp) for sp in specials])
    return [z for z,_ in hot_cnt.most_common(2)]

def two_double_cold(hist):
    omission = _zodiac_omission_map(hist)
    sorted_cold = sorted(omission.items(),key=lambda x:-x[1])
    return [sorted_cold[0][0], sorted_cold[1][0]]

def two_last2_specials(hist):
    if len(hist)<2: return ["马","蛇"]
    return [get_zodiac(hist[-1][2]), get_zodiac(hist[-2][2])]

def two_hot_special_cold_main(hist):
    specials = [sp for _,_,sp in hist[-8:]]
    hot = Counter([get_zodiac(sp) for sp in specials]).most_common(1)[0][0]
    omission = _zodiac_omission_map(hist)
    cold = max((z for z in ZODIAC_MAP if z!=hot), key=lambda z: omission[z])
    return [hot, cold]

def two_neighbor_pair(hist):
    if not hist: return ["马","蛇"]
    latest_z = get_zodiac(hist[-1][2])
    pair_z = ZODIAC_PAIR.get(latest_z, "马")
    specials = [get_zodiac(sp) for _,_,sp in hist[-8:]]
    hot = Counter(specials).most_common(1)[0][0] if specials else "蛇"
    if pair_z==hot: hot = [z for z in ZODIAC_MAP if z!=pair_z][0]
    return [pair_z, hot]

STRATEGIES_TWO = {
    "hot_cold": two_hot_cold,
    "double_hot": two_double_hot,
    "double_cold": two_double_cold,
    "last2_specials": two_last2_specials,
    "hot_special_cold_main": two_hot_special_cold_main,
    "neighbor_pair": two_neighbor_pair,
}

# ---------- 三生肖策略 (5) ----------
def three_ranked(hist):
    scores = _build_zodiac_scores_from_rows(hist, decay=0.07)
    ranked = sorted(scores.items(),key=lambda x:-x[1])
    return [z for z,_ in ranked[:3]]

def three_boosted(hist, boost):
    omission = _zodiac_omission_map(hist)
    for z in omission: omission[z] *= boost
    sorted_cold = sorted(omission.items(),key=lambda x:-x[1])
    picks = [z for z,_ in sorted_cold[:2]]
    latest_z = get_zodiac(hist[-1][2]) if hist else None
    if latest_z and latest_z not in picks: picks.append(latest_z)
    else:
        for z,_ in sorted_cold[2:]:
            if z not in picks: picks.append(z); break
    return picks[:3]

def three_momentum(hist, momentum_w):
    scores = {z:0.0 for z in ZODIAC_MAP}
    for idx,(_,nums,sp) in enumerate(hist[-12:][::-1]):
        w = momentum_w/(1.0+idx*0.2)
        for n in nums: scores[get_zodiac(n)]+=w
        scores[get_zodiac(sp)]+=w*1.5
    return [z for z,_ in sorted(scores.items(),key=lambda x:-x[1])[:3]]

def three_hybrid(hist, boost, momentum_w):
    b = three_boosted(hist, boost)
    m = three_momentum(hist, momentum_w)
    return list(dict.fromkeys(b+m))[:3]

def three_freq(hist):
    counter = Counter()
    for _,nums,sp in hist[-12:]:
        for n in nums: counter[get_zodiac(n)]+=1
        counter[get_zodiac(sp)]+=1
    return [z for z,_ in counter.most_common(3)]

STRATEGIES_THREE = {
    "ranked": three_ranked,
    "boosted": three_boosted,
    "momentum": three_momentum,
    "hybrid": three_hybrid,
    "freq": three_freq,
}

# ---------- 特肖(四只)策略 ----------
def texiao4_ranked(hist):
    """基于近期特别号生肖冷热度选择四个"""
    specials_z = [get_zodiac(sp) for _,_,sp in hist[-12:]]
    z_cnt = Counter(specials_z)
    for z in ZODIAC_MAP:
        if z not in z_cnt: z_cnt[z] = 0
    sorted_z = sorted(z_cnt.items(), key=lambda x: (x[1], x[0]))  # 冷优先
    return [z for z,_ in sorted_z[:4]]

def texiao4_hot(hist):
    """基于近期特别号生肖热度选择四个"""
    specials_z = [get_zodiac(sp) for _,_,sp in hist[-12:]]
    z_cnt = Counter(specials_z)
    for z in ZODIAC_MAP:
        if z not in z_cnt: z_cnt[z] = 0
    sorted_z = sorted(z_cnt.items(), key=lambda x: (-x[1], x[0]))  # 热优先
    return [z for z,_ in sorted_z[:4]]

def texiao4_momentum(hist):
    """基于近期特别号生肖动量选择四个"""
    scores = {z:0.0 for z in ZODIAC_MAP}
    for idx,(_,_,sp) in enumerate(hist[-12:][::-1]):
        w = 1.0/(1.0+idx*0.1)
        scores[get_zodiac(sp)] += w
    ranked = sorted(scores.items(), key=lambda x:-x[1])
    return [z for z,_ in ranked[:4]]

def texiao4_omit_break(hist):
    """选择最近未出现的生肖"""
    omission = {z: len(hist)+1 for z in ZODIAC_MAP}
    for i,(_,_,sp) in enumerate(hist[-20:][::-1], start=1):
        z = get_zodiac(sp)
        if omission[z] > i:
            omission[z] = i
    ranked = sorted(omission.items(), key=lambda x:-x[1])  # 遗漏值大的优先
    return [z for z,_ in ranked[:4]]

def texiao4_pair(hist):
    """基于最近特别号生肖的对冲生肖选择四个"""
    latest_z = get_zodiac(hist[-1][2]) if hist else "马"
    pair_z = ZODIAC_PAIR.get(latest_z, "马")
    # 选 pair 及最近冷门
    specials_z = [get_zodiac(sp) for _,_,sp in hist[-12:]]
    z_cnt = Counter(specials_z)
    for z in ZODIAC_MAP:
        if z not in z_cnt: z_cnt[z] = 0
    # 排除 pair 以外的热肖
    sorted_z = sorted(z_cnt.items(), key=lambda x: (x[1], x[0]))
    picks = [pair_z]
    for z,_ in sorted_z:
        if z not in picks: picks.append(z)
        if len(picks)==4: break
    return picks[:4]

def texiao4_mixed(hist):
    """混合冷热：2个最热+2个最冷"""
    specials_z = [get_zodiac(sp) for _,_,sp in hist[-12:]]
    z_cnt = Counter(specials_z)
    for z in ZODIAC_MAP:
        if z not in z_cnt: z_cnt[z] = 0
    hot = [z for z,_ in sorted(z_cnt.items(), key=lambda x:-x[1])[:2]]
    cold = [z for z,_ in sorted(z_cnt.items(), key=lambda x:x[1])[:2]]
    return hot + cold

STRATEGIES_TEXIAO4 = {
    "ranked": texiao4_ranked,
    "hot": texiao4_hot,
    "momentum": texiao4_momentum,
    "omit_break": texiao4_omit_break,
    "pair": texiao4_pair,
    "mixed": texiao4_mixed,
}

# ---------- 评估函数 ----------
def evaluate(issues, params, debug=False):
    total = len(issues)
    if total < 15: return -999.0, 0,0,0,0,0,0,0,0

    recent10_start = max(0, total-10)
    single_hits=0; two_hits=0; three_hits=0; texiao4_hits=0
    s_st=0; t_st=0; th_st=0; tx4_st=0
    max_s=0; max_t=0; max_th=0; max_tx4=0

    for i in range(recent10_start, total):
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        # 主号生肖
        cur_zod = set(get_zodiac(n) for n in cur_nums)
        # 特别号生肖 (用于特肖)
        cur_sp_zod = get_zodiac(cur_sp)

        # 一生肖 (主号生肖)
        s_func = STRATEGIES_SINGLE[params['single_strategy']]
        s = s_func(past, params['wsize'], params['rec_w'], params['safe_th'])
        if s in cur_zod: single_hits+=1; s_st=0
        else: s_st+=1; max_s=max(max_s,s_st)

        # 二生肖 (主号生肖)
        t_func = STRATEGIES_TWO[params['two_strategy']]
        two = t_func(past)
        if any(z in cur_zod for z in two): two_hits+=1; t_st=0
        else: t_st+=1; max_t=max(max_t,t_st)

        # 三生肖 (主号生肖，至少中2个)
        th_func = STRATEGIES_THREE[params['three_strategy']]
        if params['three_strategy'] == 'hybrid':
            three = th_func(past, params.get('three_boost', 1.0), params.get('three_momentum', 1.0))
        elif params['three_strategy'] == 'boosted':
            three = th_func(past, params.get('three_boost', 1.0))
        elif params['three_strategy'] == 'momentum':
            three = th_func(past, params.get('three_momentum', 1.0))
        else:
            three = th_func(past)
        hit_count = sum(1 for z in three if z in cur_zod)
        if hit_count >= 2: three_hits+=1; th_st=0
        else: th_st+=1; max_th=max(max_th,th_st)

        # 四只特肖 (特别号生肖，至少中1个)
        tx4_func = STRATEGIES_TEXIAO4[params['texiao4_strategy']]
        four_z = tx4_func(past)
        if cur_sp_zod in four_z: texiao4_hits+=1; tx4_st=0
        else: tx4_st+=1; max_tx4=max(max_tx4,tx4_st)

    n = total - recent10_start
    if n==0: return -999.0,0,0,0,0,0,0,0,0
    r1 = single_hits/n
    r2 = two_hits/n
    r3 = three_hits/n
    r4 = texiao4_hits/n

    # 评分权重：一生肖35% + 二生肖35% + 三生肖20% + 特肖10%
    score = r1 * 0.35 + r2 * 0.35 + r3 * 0.20 + r4 * 0.10

    return score, r1, r2, r3, r4, max_s, max_t, max_th, max_tx4

def objective(trial, issues):
    p = {
        'single_strategy': trial.suggest_categorical('single_strategy', list(STRATEGIES_SINGLE.keys())),
        'two_strategy': trial.suggest_categorical('two_strategy', list(STRATEGIES_TWO.keys())),
        'three_strategy': trial.suggest_categorical('three_strategy', list(STRATEGIES_THREE.keys())),
        'texiao4_strategy': trial.suggest_categorical('texiao4_strategy', list(STRATEGIES_TEXIAO4.keys())),
        'wsize': trial.suggest_int('wsize', 2, 20),
        'rec_w': trial.suggest_float('rec_w', 0.1, 4.0),
        'safe_th': trial.suggest_float('safe_th', 0.4, 2.5),
        'three_boost': trial.suggest_float('three_boost', 0.3, 6.0),
        'three_momentum': trial.suggest_float('three_momentum', 0.5, 3.0),
    }
    score, _,_,_,_,_,_,_,_ = evaluate(issues, p)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='hk_marksix.db')
    parser.add_argument('--trials', type=int, default=5000)
    parser.add_argument('--recent', type=int, default=120)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    import optuna.logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    conn = connect_db(args.db)
    issues = load_issues(conn, recent=args.recent)
    conn.close()
    if len(issues) < 20:
        print("数据不足"); sys.exit(1)

    study = optuna.create_study(
        direction='maximize',
        study_name='hk_strategy_evolution',
        storage='sqlite:///optuna_hk_stable.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(lambda t: objective(t, issues), n_trials=args.trials, show_progress_bar=False)

    best_p = study.best_params
    score, r1, r2, r3, r4, ms1, ms2, ms3, ms4 = evaluate(issues, best_p, debug=args.debug)

    print(f"\n最优策略: 单:{best_p['single_strategy']} 二:{best_p['two_strategy']} 三:{best_p['three_strategy']} 特肖:{best_p['texiao4_strategy']}")
    print(f"近10期: 一生肖={r1:.3f}(连空{ms1}) 二肖={r2:.3f}(连空{ms2}) 三肖(中2)={r3:.3f}(连空{ms3}) 特肖(中1)={r4:.3f}(连空{ms4})")

    with open("best_params_hk.json", "w") as f:
        json.dump(best_p, f, indent=2)

    # 达标标准：一生肖≥90%, 二肖≥90%, 三肖≥90%, 特肖≥100%(即连空0)
    if r1 >= 0.90 and r2 >= 0.90 and r3 >= 0.90 and r4 >= 1.0 and max(ms1,ms2,ms3,ms4) <= 0:
        print("🎉 达标！"); sys.exit(0)
    else:
        print("未达标，继续搜索"); sys.exit(1)

if __name__=="__main__":
    main()