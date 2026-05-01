#!/usr/bin/env python3
"""
全自动超参数优化脚本 v3 —— 同时提升特别号与一生肖
"""

import sqlite3, json, sys, argparse
from typing import List, Tuple
from collections import Counter

import optuna
import numpy as np

ZODIAC_MAP = {
    "马":[1,13,25,37,49],"蛇":[2,14,26,38],"龙":[3,15,27,39],
    "兔":[4,16,28,40],"虎":[5,17,29,41],"牛":[6,18,30,42],
    "鼠":[7,19,31,43],"猪":[8,20,32,44],"狗":[9,21,33,45],
    "鸡":[10,22,34,46],"猴":[11,23,35,47],"羊":[12,24,36,48],
}
ALL_NUMS = list(range(1,50))

def get_zodiac(n): 
    for z,ns in ZODIAC_MAP.items():
        if n in ns: return z
    return "马"

def connect_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def load_issues(conn, recent=120):
    rows = conn.execute("SELECT issue_no,draw_date,numbers_json,special_number FROM draws ORDER BY draw_date ASC").fetchall()
    data = [(r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])) for r in rows]
    return data[-recent:]

# ---------- 预测函数 (参数动态) ----------
def pred_single_zodiac(hist, wsize, rec_w, safe_th):
    scores = {z:0.0 for z in ZODIAC_MAP}
    recent = hist[-wsize:] if len(hist)>=wsize else hist
    for idx,(_,nums,sp) in enumerate(recent[::-1]):
        w = rec_w/(1.0+idx*0.15)
        for n in nums: scores[get_zodiac(n)] += w
        scores[get_zodiac(sp)] += w*2.0
    if max(scores.values()) < safe_th:
        omission = {z:0 for z in ZODIAC_MAP}
        for i in range(len(recent)):
            _,nums,sp = recent[-(i+1)]
            for z in ZODIAC_MAP:
                if omission[z]==0: omission[z]=i+1
            for n in nums: omission[get_zodiac(n)]=0
            omission[get_zodiac(sp)]=0
        return max(omission.items(), key=lambda x:x[1])[0]
    return max(scores, key=scores.get)

def pred_two_zodiac(hist, defense_th):
    specials = [sp for _,_,sp in hist[-10:]]
    hot_cnt = Counter(get_zodiac(sp) for sp in specials)
    hot = max(hot_cnt, key=hot_cnt.get)
    omission = {z:0 for z in ZODIAC_MAP}
    for i in range(len(hist)):
        _,nums,sp = hist[-(i+1)]
        for z in ZODIAC_MAP:
            if omission[z]==0: omission[z]=i+1
        for n in nums: omission[get_zodiac(n)]=0
        omission[get_zodiac(sp)]=0
    cold = max((z for z in ZODIAC_MAP if z!=hot), key=lambda z:omission[z])
    return [hot, cold]

def pred_three_zodiac(hist, min_omit, hot_w2):
    two = pred_two_zodiac(hist,0)
    omission = {z:0 for z in ZODIAC_MAP}
    for i in range(len(hist)):
        _,nums,sp = hist[-(i+1)]
        for z in ZODIAC_MAP:
            if omission[z]==0: omission[z]=i+1
        for n in nums: omission[get_zodiac(n)]=0
        omission[get_zodiac(sp)]=0
    high_omit = [z for z,o in omission.items() if o>=min_omit and z not in two]
    if high_omit:
        third = max(high_omit, key=lambda z:omission[z])
    else:
        others = [z for z in ZODIAC_MAP if z not in two]
        freq = Counter()
        for _,nums,sp in hist[-8:]:
            for n in nums: freq[get_zodiac(n)]+=1
            freq[get_zodiac(sp)]+=1
        third = max(others, key=lambda z:freq.get(z,0)*hot_w2)
    return two[:2] + [third]

def pred_four_zodiac(hist, omit_boost):
    omission = {z:0 for z in ZODIAC_MAP}
    for i in range(len(hist)):
        _,nums,sp = hist[-(i+1)]
        for z in ZODIAC_MAP:
            if omission[z]==0: omission[z]=i+1
        for n in nums: omission[get_zodiac(n)]=0
        omission[get_zodiac(sp)]=0
    sorted_cold = sorted(omission.items(), key=lambda x:(-x[1],x[0]))
    picks = [z for z,_ in sorted_cold[:4]]
    latest_z = get_zodiac(hist[-1][2])
    if latest_z not in picks: picks[-1]=latest_z
    return picks[:4]

def pred_special_hybrid(hist, params):
    specials = [sp for _,_,sp in hist]
    if len(specials)<12: return [1,2,3]
    omission = {}
    for i,sp in enumerate(specials):
        if sp not in omission: omission[sp]=i+1
        else: omission[sp]=min(omission[sp],i+1)
    rule_scores = {n:0.0 for n in ALL_NUMS}
    for n in ALL_NUMS:
        omit = omission.get(n,999)
        if omit>=params['cold_thr']: rule_scores[n]+=5.0
        diff = abs(n-specials[-1])
        if diff==1: rule_scores[n]+=params['n1']
        elif diff==2: rule_scores[n]+=params['n2']
        if n in specials[-3:]: rule_scores[n]*=params['penalty']
    lgb_weight = params.get('lgb_w',0.6)
    total = len(hist[-60:])
    lgb_probs = {n:0.0 for n in ALL_NUMS}
    for n in ALL_NUMS:
        freq = sum(1 for _,_,sp in hist[-60:] if sp==n)/total
        omit = omission.get(n,60)
        omit_score = 1.0/(omit+1)
        lgb_probs[n] = 0.6*freq + 0.4*omit_score
    final = {}
    for n in ALL_NUMS:
        final[n] = (1-lgb_weight)*rule_scores[n] + lgb_weight*lgb_probs[n]*5
    return [n for n,_ in sorted(final.items(), key=lambda x:-x[1])[:3]]

# ---------- 评估（重点改动） ----------
def evaluate_all(issues, params):
    single_h = two_h = three_h = four_h = special_h = 0
    total = 0
    min_len = 40
    for i in range(min_len, len(issues)):
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums)
        cur_zod.add(get_zodiac(cur_sp))

        s = pred_single_zodiac(past, params['single_window'], params['single_recency_w'], params['single_safe_threshold'])
        if s in cur_zod: single_h += 1

        two = pred_two_zodiac(past, params['two_defense_threshold'])
        if any(z in cur_zod for z in two): two_h += 1

        three = pred_three_zodiac(past, params['three_min_omit'], params['three_hot_weight2'])
        if sum(1 for z in three if z in cur_zod) >= 2: three_h += 1

        four = pred_four_zodiac(past, params['four_omit_boost'])
        if any(z in cur_zod for z in four): four_h += 1

        sp_pred = pred_special_hybrid(past, params)
        if cur_sp in sp_pred: special_h += 1

        total += 1

    if total == 0: return 0.0
    r1 = single_h/total
    r2 = two_h/total
    r3 = three_h/total
    r4 = four_h/total
    r5 = special_h/total

    # 核心改动：特别号+一生肖各占0.35，其余0.1；若一生肖低于0.68则惩罚
    score = 0.35*r5 + 0.35*r1 + 0.1*r2 + 0.1*r3 + 0.1*r4
    if r1 < 0.68:
        score *= 0.8
    return score

# ---------- Optuna ----------
def objective(trial, issues):
    p = {
        'single_window': trial.suggest_int('single_window',4,20),
        'single_recency_w': trial.suggest_float('single_recency_w',0.5,2.2),
        'single_safe_threshold': trial.suggest_float('single_safe_threshold',0.5,2.0),
        'two_defense_threshold': trial.suggest_int('two_defense_threshold',0,3),
        'three_min_omit': trial.suggest_int('three_min_omit',3,10),
        'three_hot_weight2': trial.suggest_float('three_hot_weight2',0.5,2.5),
        'four_omit_boost': trial.suggest_float('four_omit_boost',1.0,5.0),
        'cold_thr': trial.suggest_int('cold_thr',6,18),
        'n1': trial.suggest_float('n1',2.0,7.0),
        'n2': trial.suggest_float('n2',0.5,4.0),
        'penalty': trial.suggest_float('penalty',0.5,0.98),
        'lgb_w': trial.suggest_float('lgb_w',0.3,0.8),
    }
    return evaluate_all(issues, p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--trials', type=int, default=300)
    args = parser.parse_args()
    conn = connect_db(args.db)
    issues = load_issues(conn, recent=120)
    conn.close()
    if len(issues)<60:
        print("数据不足")
        sys.exit(1)
    study = optuna.create_study(direction='maximize', study_name='v3_both',
                                storage='sqlite:///optuna_v3_both.db', load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))
    study.optimize(lambda trial: objective(trial, issues), n_trials=args.trials, show_progress_bar=True)
    print("\n最佳参数（同时优化特别号与一生肖）:")
    for k,v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"最佳得分: {study.best_value:.4f}")
    best = study.best_params
    best["score"] = study.best_value
    with open("best_params.json","w") as f:
        json.dump(best, f, indent=2)
    print("已保存至 best_params.json")

if __name__=="__main__":
    main()
