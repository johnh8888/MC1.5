#!/usr/bin/env python3
"""
澳门六合彩全自动策略进化优化器（最终稳定版）
移除 random_baseline，权重 0.20/0.20/0.20/0.40，带微小扰动
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

# ---------- 基础工具 ----------
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

# ========== 一生肖策略 (6) ==========
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

# ========== 二生肖策略 (6) ==========
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

# ========== 四生肖策略 (5) ==========
def four_boosted(hist, four_boost):
    omission = {z:0 for z in ZODIAC_MAP}
    specials = [sp for _,_,sp in hist]
    for i,sp in enumerate(specials[::-1]):
        z = get_zodiac(sp)
        if omission[z]==0: omission[z]=i+1
    for z in omission: omission[z]*=four_boost
    sorted_cold = sorted(omission.items(),key=lambda x:-x[1])
    picks = [z for z,_ in sorted_cold[:3]]
    latest_z = get_zodiac(specials[-1]) if specials else None
    if latest_z and latest_z not in picks:
        picks.append(latest_z)
    else:
        for z,_ in sorted_cold[3:]:
            if z not in picks: picks.append(z);break
    return picks[:4]

def four_momentum(hist, momentum_w):
    scores = {z:0.0 for z in ZODIAC_MAP}
    for idx,(_,nums,sp) in enumerate(hist[-12:][::-1]):
        w = momentum_w/(1.0+idx*0.2)
        for n in nums: scores[get_zodiac(n)]+=w
        scores[get_zodiac(sp)]+=w*1.5
    return [z for z,_ in sorted(scores.items(),key=lambda x:-x[1])[:4]]

def four_hybrid(hist, four_boost, momentum_w):
    b = four_boosted(hist, four_boost)
    m = four_momentum(hist, momentum_w)
    union = list(dict.fromkeys(b+m))
    return union[:4]

def four_top4_freq(hist, recent_n=12):
    counter = Counter()
    for _,nums,sp in hist[-recent_n:]:
        for n in nums: counter[get_zodiac(n)]+=1
        counter[get_zodiac(sp)]+=1
    return [z for z,_ in counter.most_common(4)]

def four_cold_main_only(hist, recent_n=20):
    omission = {z:0 for z in ZODIAC_MAP}
    for i,(_,nums,sp) in enumerate(hist[-recent_n:]):
        for z in ZODIAC_MAP: omission[z]=omission.get(z,i+1)
        for n in nums: omission[get_zodiac(n)]=0
    sorted_cold = sorted(omission.items(),key=lambda x:-x[1])
    return [z for z,_ in sorted_cold[:4]]

STRATEGIES_FOUR = {
    "boosted": four_boosted,
    "momentum": four_momentum,
    "hybrid": four_hybrid,
    "top4_freq": four_top4_freq,
    "cold_main_only": four_cold_main_only,
}

# ========== 特别号策略 (7) 与主脚本完全一致 ==========
def special_cold_neighbor(hist, zodiac_pool, params, top_n=3):
    recent_specials = [row[2] for row in hist[-12:][::-1]]
    if not recent_specials: return [1,2,3]
    latest = recent_specials[0]
    omission = {}
    for i,sp in enumerate(recent_specials):
        if sp not in omission: omission[sp]=i+1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z,[])))
    if not candidates: return [1,2,3]
    cold_th = params['cold_threshold']
    nb1 = params['neighbor_1_bonus']
    nb2 = params['neighbor_2_bonus']
    pen = params['penalty_coeff']
    lw = params['lgb_weight']
    ob = params['four_omit_boost']
    picks = sorted([n for n in candidates if omission.get(n,20)>=cold_th],key=lambda n:omission.get(n,20),reverse=True)[:2]
    while len(picks)<2:
        remaining = [n for n in candidates if n not in picks]
        if not remaining: break
        picks.append(max(remaining,key=lambda n:omission.get(n,20)))
    if len(picks)<top_n:
        nb = [n for n in candidates if abs(n-latest)==1 and n not in picks]
        if nb: picks.append(max(nb,key=lambda n:omission.get(n,20)+nb1))
    if len(picks)<top_n:
        nb2l = [n for n in candidates if abs(n-latest)==2 and n not in picks]
        if nb2l: picks.append(max(nb2l,key=lambda n:omission.get(n,20)+nb2))
    while len(picks)<top_n:
        rest = sorted([n for n in candidates if n not in picks],key=lambda n:omission.get(n,20),reverse=True)
        if rest: picks.append(rest[0])
        else: break
    scored = []
    for n in picks:
        score = omission.get(n,20)*lw
        if n in recent_specials[:3]: score*=pen
        if omission.get(n,20)>=cold_th: score+=ob
        scored.append((n,score))
    scored.sort(key=lambda x:-x[1])
    return [n for n,_ in scored[:top_n]]

def special_tail_focus(hist, zodiac_pool, params, top_n=3):
    recent_specials = [row[2] for row in hist[-12:][::-1]]
    if not recent_specials: return [1,2,3]
    latest = recent_specials[0]
    tail_counter = Counter()
    for row in hist[-12:]:
        for n in row[1]: tail_counter[n%10]+=1
        tail_counter[row[2]%10]+=3
    hot_tails = [t for t,_ in tail_counter.most_common(6)]
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z,[])))
    if not candidates: return [1,2,3]
    picks = []
    last_tail = latest%10
    for t in [last_tail,(last_tail+1)%10,(last_tail-1)%10,*hot_tails]:
        tail_nums = [n for n in candidates if n%10==t and n not in picks]
        if tail_nums:
            picks.append(max(tail_nums,key=lambda n: sum(1 for row in hist[-20:] if n in row[1] or n==row[2])))
        if len(picks)>=top_n: break
    while len(picks)<top_n:
        rest = [n for n in candidates if n not in picks]
        if not rest: break
        picks.append(rest[0])
    return picks[:top_n]

def special_omission_only(hist, zodiac_pool, params, top_n=3):
    recent_specials = [row[2] for row in hist[-12:][::-1]]
    if not recent_specials: return [1,2,3]
    omission = {}
    for i,sp in enumerate(recent_specials):
        if sp not in omission: omission[sp]=i+1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z,[])))
    if not candidates: return [1,2,3]
    return sorted(candidates,key=lambda n:omission.get(n,30),reverse=True)[:top_n]

def special_zone_bias(hist, zodiac_pool, params, top_n=3):
    recent_specials = [row[2] for row in hist[-12:][::-1]]
    if not recent_specials: return [1,2,3]
    zones = {"low":0,"mid":0,"high":0}
    for sp in recent_specials:
        if sp<=19: zones["low"]+=1
        elif sp<=39: zones["mid"]+=1
        else: zones["high"]+=1
    target_zone = min(zones,key=zones.get)
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z,[])))
    if target_zone=="low": candidates = [n for n in candidates if n<=19]
    elif target_zone=="mid": candidates = [n for n in candidates if 20<=n<=39]
    else: candidates = [n for n in candidates if n>=40]
    if not candidates: return [1,2,3]
    omission = {}
    for i,sp in enumerate(recent_specials):
        if sp not in omission: omission[sp]=i+1
    return sorted(candidates,key=lambda n:omission.get(n,30),reverse=True)[:top_n]

def special_neighbor_tail(hist, zodiac_pool, params, top_n=3):
    recent_specials = [row[2] for row in hist[-12:][::-1]]
    if not recent_specials: return [1,2,3]
    latest = recent_specials[0]
    omission = {}
    for i,sp in enumerate(recent_specials):
        if sp not in omission: omission[sp]=i+1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z,[])))
    picks = []
    nb1 = [n for n in candidates if abs(n-latest)==1 and n not in picks]
    picks.extend(sorted(nb1,key=lambda n:omission.get(n,30),reverse=True)[:1])
    tail = [n for n in candidates if n%10==latest%10 and n not in picks]
    picks.extend(sorted(tail,key=lambda n:omission.get(n,30),reverse=True)[:1])
    while len(picks)<top_n:
        rest = sorted([n for n in candidates if n not in picks],key=lambda n:omission.get(n,30),reverse=True)
        if rest: picks.append(rest[0])
        else: break
    return picks[:top_n]

def special_mixed_2cold1hot(hist, zodiac_pool, params, top_n=3):
    recent_specials = [row[2] for row in hist[-12:][::-1]]
    if not recent_specials: return [1,2,3]
    omission = {}
    for i,sp in enumerate(recent_specials):
        if sp not in omission: omission[sp]=i+1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z,[])))
    cold = sorted(candidates,key=lambda n:omission.get(n,30),reverse=True)[:2]
    hot = sorted(candidates,key=lambda n: recent_specials.count(n), reverse=True)
    hot = [n for n in hot if n not in cold][:1]
    return cold + hot

def special_omit_break(hist, zodiac_pool, params, top_n=3):
    recent_specials = [row[2] for row in hist[-12:][::-1]]
    if not recent_specials: return [1,2,3]
    latest = recent_specials[0]
    omission = {}
    for i,sp in enumerate(recent_specials):
        if sp not in omission: omission[sp]=i+1
    candidates = list(set(n for z in zodiac_pool for n in ZODIAC_MAP.get(z,[])))
    extreme = [n for n in candidates if omission.get(n,30)>=20]
    if extreme: return extreme[:top_n]
    nb1 = [n for n in candidates if abs(n-latest)==1]
    if nb1: return nb1[:top_n]
    return sorted(candidates,key=lambda n:omission.get(n,30),reverse=True)[:top_n]

# 移除 random_baseline
STRATEGIES_SPECIAL = {
    "cold_neighbor": special_cold_neighbor,
    "tail_focus": special_tail_focus,
    "omission_only": special_omission_only,
    "zone_bias": special_zone_bias,
    "neighbor_tail": special_neighbor_tail,
    "mixed_2cold1hot": special_mixed_2cold1hot,
    "omit_break": special_omit_break,
}

# ========== 评估函数 (权重调整) ==========
def evaluate(issues, params, debug=False):
    total = len(issues)
    if total < 15: return -999.0, 0,0,0,0,0,0,0
    recent10_start = max(0, total-10)
    single_hits=0; two_hits=0; four_hits=0; special_hits=0
    s_st=0; t_st=0; f_st=0; sp_st=0
    max_s=0; max_t=0; max_f=0; max_sp=0

    for i in range(recent10_start, total):
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums) | {get_zodiac(cur_sp)}

        # 一生肖
        s_func = STRATEGIES_SINGLE[params['single_strategy']]
        s = s_func(past, params['wsize'], params['rec_w'], params['safe_th'])
        if s in cur_zod: single_hits+=1; s_st=0
        else: s_st+=1; max_s=max(max_s,s_st)

        # 二生肖
        t_func = STRATEGIES_TWO[params['two_strategy']]
        two = t_func(past)
        if any(z in cur_zod for z in two): two_hits+=1; t_st=0
        else: t_st+=1; max_t=max(max_t,t_st)

        # 四生肖 (针对特别号生肖)
        f_strat = params['four_strategy']
        if f_strat == "boosted":
            four = four_boosted(past, params['four_boost'])
        elif f_strat == "momentum":
            four = four_momentum(past, params.get('momentum_w',1.0))
        elif f_strat == "hybrid":
            four = four_hybrid(past, params['four_boost'], params.get('momentum_w',1.0))
        elif f_strat == "top4_freq":
            four = four_top4_freq(past)
        elif f_strat == "cold_main_only":
            four = four_cold_main_only(past)
        else:
            four = four_boosted(past, params['four_boost'])
        # 四生肖命中标准：实际特别号生肖在预测的四肖中
        actual_zod = get_zodiac(cur_sp)
        if actual_zod in four: four_hits+=1; f_st=0
        else: f_st+=1; max_f=max(max_f,f_st)

        # 特别号 (3码精选)
        base_four = four  # 用四肖作为基础
        last3_zodiacs = [get_zodiac(r[2]) for r in past[-3:]]
        enh_pool = list(dict.fromkeys(base_four + last3_zodiacs))
        while len(enh_pool) < 4:
            for z in ZODIAC_MAP:
                if z not in enh_pool: enh_pool.append(z); break
        sp_func = STRATEGIES_SPECIAL[params['special_strategy']]
        sp_picks = sp_func(past, enh_pool, params, 3)
        if cur_sp in sp_picks: special_hits+=1; sp_st=0
        else: sp_st+=1; max_sp=max(max_sp,sp_st)

        if debug and i>=recent10_start:
            print(f"[调试] {issues[i][0]} 单:{params['single_strategy']} 二:{params['two_strategy']} "
                  f"四:{params['four_strategy']} 特:{params['special_strategy']} | "
                  f"预测特号:{sp_picks} 实际:{cur_sp} {'√' if cur_sp in sp_picks else '×'}")

    n = total - recent10_start
    if n==0: return -999.0,0,0,0,0,0,0,0
    r1 = single_hits/n
    r2 = two_hits/n
    r4 = four_hits/n
    rsp = special_hits/n
    max_strk = max(max_s, max_t, max_f, max_sp)
    streak_factor = 1.0
    if max_strk>=4: streak_factor=0.2
    elif max_strk==3: streak_factor=0.5
    elif max_strk==2: streak_factor=0.8

    # 调整权重：四肖和特别号权重提升
    score = r1*0.20 + r2*0.20 + r4*0.20 + rsp*0.40
    if r1<0.70: score*=0.80
    if r2<0.80: score*=0.80
    if r4<0.80: score*=0.90    # 放宽四肖阈值
    if rsp<0.50: score*=0.90
    return score*streak_factor, r1, r2, r4, rsp, max_s, max_t, max_f, max_sp

def objective(trial, issues):
    p = {
        'single_strategy': trial.suggest_categorical('single_strategy', list(STRATEGIES_SINGLE.keys())),
        'two_strategy': trial.suggest_categorical('two_strategy', list(STRATEGIES_TWO.keys())),
        'four_strategy': trial.suggest_categorical('four_strategy', list(STRATEGIES_FOUR.keys())),
        'special_strategy': trial.suggest_categorical('special_strategy', list(STRATEGIES_SPECIAL.keys())),
        'wsize': trial.suggest_int('wsize', 2, 20),
        'rec_w': trial.suggest_float('rec_w', 0.1, 4.0),
        'safe_th': trial.suggest_float('safe_th', 0.4, 2.5),
        'four_boost': trial.suggest_float('four_boost', 0.3, 6.0),
        'momentum_w': trial.suggest_float('momentum_w', 0.5, 3.0),
        'cold_threshold': trial.suggest_int('cold_threshold', 8, 18),
        'neighbor_1_bonus': trial.suggest_float('neighbor_1_bonus', 2.0, 10.0),
        'neighbor_2_bonus': trial.suggest_float('neighbor_2_bonus', 0.0, 5.0),
        'penalty_coeff': trial.suggest_float('penalty_coeff', 0.5, 1.0),
        'lgb_weight': trial.suggest_float('lgb_weight', 0.3, 1.0),
        'four_omit_boost': trial.suggest_float('four_omit_boost', 1.0, 5.0),
    }
    score, _,_,_,_,_,_,_,_ = evaluate(issues, p)
    # 微小扰动，打破平局
    score += random.uniform(-0.03, 0.03)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
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
        study_name='macau_strategy_evolution',
        storage='sqlite:///optuna_macau_stable.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(lambda t: objective(t, issues), n_trials=args.trials, show_progress_bar=False)

    best_p = study.best_params
    score, r1, r2, r4, rsp, ms1, ms2, ms4, mssp = evaluate(issues, best_p, debug=args.debug)
    print(f"\n最优策略: 单:{best_p['single_strategy']} 二:{best_p['two_strategy']} "
          f"四:{best_p['four_strategy']} 特:{best_p['special_strategy']}")
    print(f"近10期: 一生肖={r1:.3f}(连空{ms1}) 二肖={r2:.3f}(连空{ms2}) "
          f"四肖={r4:.3f}(连空{ms4}) 特别号={rsp:.3f}(连空{mssp})")
    with open("best_params_zodiac.json", "w") as f:
        json.dump(best_p, f, indent=2)

    if r1>=0.70 and r2>=0.80 and r4>=0.85 and rsp>=0.50 and max(ms1,ms2,ms4,mssp)<=1:
        print("🎉 达标！"); sys.exit(0)
    else:
        print("未达标，继续搜索"); sys.exit(1)

if __name__=="__main__":
    main()
