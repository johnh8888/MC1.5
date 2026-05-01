#!/usr/bin/env python3
"""
全自动超参数优化脚本 v2（近期120期 + 新参数 + LightGBM 融合模拟）
使用方法：
    python hyper_optimize.py --db newmacau_marksix.db --trials 300
"""

import sqlite3
import json
import sys
import argparse
from typing import List, Tuple, Dict
from collections import Counter

import optuna
import numpy as np

# ---------- 基础配置 ----------
ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}
ALL_NUMBERS = list(range(1, 50))

def get_zodiac(n: int) -> str:
    for z, nums in ZODIAC_MAP.items():
        if n in nums:
            return z
    return "马"

def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_all_issues(conn, recent=120) -> List[Tuple[str, List[int], int]]:
    """从数据库加载开奖数据，仅保留最近 recent 期"""
    rows = conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
    data = []
    for r in rows:
        data.append((r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])))
    return data[-recent:]  # 只使用最近 recent 期

# ---------- 预测函数（新增参数可调） ----------

def predict_single_zodiac(history: list, window: int, recency_w: float, safe_threshold: float) -> str:
    """一生肖预测（支持保底切换）"""
    scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = history[-window:] if len(history) >= window else history
    for idx, (_, nums, sp) in enumerate(recent[::-1]):
        w = recency_w / (1.0 + idx * 0.15)
        for n in nums:
            scores[get_zodiac(n)] += w
        scores[get_zodiac(sp)] += w * 2.0

    # 如果近期命中率偏低（模拟），则保底切换到遗漏最大的生肖
    # 这里的“近期命中率”用最高分判断：若最高分过低，可能进入冷态
    if max(scores.values()) < safe_threshold:
        omission = {z: 0 for z in ZODIAC_MAP}
        for i in range(len(recent)):
            _, nums, sp = recent[-(i+1)]
            for z in ZODIAC_MAP:
                if omission[z] == 0:
                    omission[z] = i + 1
            for n in nums:
                omission[get_zodiac(n)] = 0
            omission[get_zodiac(sp)] = 0
        return max(omission.items(), key=lambda x: x[1])[0]
    return max(scores, key=scores.get)

def predict_two_zodiac(history: list, defense_threshold: int) -> List[str]:
    """二生肖预测（支持防守切换）：若近N期未中，切换为全冷模式"""
    specials = [sp for _, _, sp in history[-10:]]
    hot_counter = Counter([get_zodiac(sp) for sp in specials])
    hot = max(hot_counter, key=hot_counter.get)

    # 模拟防守阈值：若最近一期预测未中（这里用上次预测未中次数>=defense_threshold则防守）
    # 简化：始终使用一热一冷，但冷号的权重由外部冷号权重参数控制（此处暂固定）
    omission = {z: 0 for z in ZODIAC_MAP}
    for i in range(len(history)):
        _, nums, sp = history[-(i+1)]
        for z in ZODIAC_MAP:
            if omission[z] == 0:
                omission[z] = i + 1
        for n in nums:
            omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    cold = max((z for z in ZODIAC_MAP if z != hot), key=lambda z: omission[z])
    return [hot, cold]

def predict_three_zodiac(history: list, min_omit: int, hot_weight2: float) -> List[str]:
    """三生肖预测：前二同二生肖，第三位优先遗漏>=min_omit，否则用热号加权"""
    two = predict_two_zodiac(history, 0)  # 二生肖暂时忽略防守参数
    omission = {z: 0 for z in ZODIAC_MAP}
    for i in range(len(history)):
        _, nums, sp = history[-(i+1)]
        for z in ZODIAC_MAP:
            if omission[z] == 0:
                omission[z] = i + 1
        for n in nums:
            omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    high_omit = [z for z, o in omission.items() if o >= min_omit and z not in two]
    if high_omit:
        third = max(high_omit, key=lambda z: omission[z])
    else:
        # 无高遗漏，选热号第二位（排除已选的）
        all_but_two = [z for z in ZODIAC_MAP if z not in two]
        # 简单按频率
        freq = Counter()
        for _, nums, sp in history[-8:]:
            for n in nums:
                freq[get_zodiac(n)] += 1
            freq[get_zodiac(sp)] += 1
        third = max(all_but_two, key=lambda z: freq.get(z, 0) * hot_weight2)
    return two[:2] + [third]

def predict_four_zodiac(history: list, omit_boost: float) -> List[str]:
    """特别生肖：冷号策略，遗漏加权"""
    omission = {z: 0 for z in ZODIAC_MAP}
    for i in range(len(history)):
        _, nums, sp = history[-(i+1)]
        for z in ZODIAC_MAP:
            if omission[z] == 0:
                omission[z] = i + 1
        for n in nums:
            omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    sorted_cold = sorted(omission.items(), key=lambda x: (-x[1], x[0]))
    picks = [z for z, _ in sorted_cold[:4]]
    # 最近一期特别号生肖加入替换
    latest_z = get_zodiac(history[-1][2])
    if latest_z not in picks:
        picks[-1] = latest_z
    return picks[:4]

def predict_special_numbers_hybrid(history: list, params: dict) -> List[int]:
    """
    特别号精选（混合规则 + 模拟 LightGBM 概率）
    规则部分：双冷一邻
    模拟 LGB：使用历史频率+遗漏构建伪概率
    """
    specials = [sp for _, _, sp in history]
    if len(specials) < 12:
        return [1, 2, 3]

    # 计算全局遗漏
    omission = {}
    for i, sp in enumerate(specials):
        if sp not in omission:
            omission[sp] = i + 1
        else:
            omission[sp] = min(omission[sp], i + 1)

    # ---- 规则得分 ----
    rule_scores = {n: 0.0 for n in ALL_NUMBERS}
    for n in ALL_NUMBERS:
        # 冷号分数
        omit = omission.get(n, 999)
        if omit >= params['cold_threshold']:
            rule_scores[n] += 5.0
        # 邻号分数
        latest = specials[-1]
        diff = abs(n - latest)
        if diff == 1:
            rule_scores[n] += params['neighbor1']
        elif diff == 2:
            rule_scores[n] += params['neighbor2']
        # 近三期惩罚
        if n in specials[-3:]:
            rule_scores[n] *= params['recent_penalty']

    # ---- 模拟 LightGBM 概率（基于频率和遗漏） ----
    # 这是一个快速近似，避免真正训练模型
    lgb_probs = {n: 0.0 for n in ALL_NUMBERS}
    total_issues = len(history[-60:])  # 近60期
    for n in ALL_NUMBERS:
        # 出现频率
        freq = sum(1 for _, _, sp in history[-60:] if sp == n) / total_issues
        # 遗漏倒数映射
        omit = omission.get(n, 60)
        omit_score = 1.0 / (omit + 1)
        lgb_probs[n] = 0.6 * freq + 0.4 * omit_score

    # ---- 融合得分 ----
    final_scores = {}
    lgb_weight = params.get('lgb_weight', 0.6)
    rule_weight = 1.0 - lgb_weight
    for n in ALL_NUMBERS:
        final_scores[n] = rule_weight * rule_scores[n] + lgb_weight * lgb_probs[n] * 5  # 调整量级

    # 选TOP3
    sorted_nums = sorted(final_scores.items(), key=lambda x: -x[1])
    return [n for n, _ in sorted_nums[:3]]

# ---------- 综合评估（新增指标） ----------

def evaluate_all(history_issues: list, params: dict) -> float:
    single_hits = two_hits = three_hits = four_hits = special_hits = 0
    total = 0
    min_len = 40  # 至少40期历史才能开始预测

    for i in range(min_len, len(history_issues)):
        past = history_issues[:i]
        cur_nums, cur_sp = history_issues[i][1], history_issues[i][2]
        cur_zodiacs = set(get_zodiac(n) for n in cur_nums)
        cur_zodiacs.add(get_zodiac(cur_sp))

        # 一生肖
        single = predict_single_zodiac(past, params['single_window'],
                                       params['single_recency_w'],
                                       params['single_safe_threshold'])
        if single in cur_zodiacs:
            single_hits += 1

        # 二生肖
        two = predict_two_zodiac(past, params['two_defense_threshold'])
        if any(z in cur_zodiacs for z in two):
            two_hits += 1

        # 三生肖 (至少中2)
        three = predict_three_zodiac(past, params['three_min_omit'],
                                     params['three_hot_weight2'])
        if sum(1 for z in three if z in cur_zodiacs) >= 2:
            three_hits += 1

        # 特别生肖 (中1)
        four = predict_four_zodiac(past, params['four_omit_boost'])
        if any(z in cur_zodiacs for z in four):
            four_hits += 1

        # 特别号精选 (双冷一邻 + LGB融合)
        sp_pred = predict_special_numbers_hybrid(past, params)
        if cur_sp in sp_pred:
            special_hits += 1

        total += 1

    if total == 0:
        return 0.0

    # 计算各模块命中率，然后按权重综合（可调整权重）
    r1 = single_hits / total
    r2 = two_hits / total
    r3 = three_hits / total
    r4 = four_hits / total
    r5 = special_hits / total
    # 权重：特别号权重稍高，其余平均
    weighted = 0.1 * r1 + 0.15 * r2 + 0.15 * r3 + 0.2 * r4 + 0.4 * r5
    return weighted

# ---------- Optuna 目标函数（参数空间扩大） ----------

def objective(trial: optuna.Trial, all_issues: list) -> float:
    params = {
        # 一生肖
        'single_window': trial.suggest_int('single_window', 4, 20),
        'single_recency_w': trial.suggest_float('single_recency_w', 0.5, 2.2),
        'single_safe_threshold': trial.suggest_float('single_safe_threshold', 0.5, 2.0),
        # 二生肖
        'two_defense_threshold': trial.suggest_int('two_defense_threshold', 0, 3),
        # 三生肖
        'three_min_omit': trial.suggest_int('three_min_omit', 3, 10),
        'three_hot_weight2': trial.suggest_float('three_hot_weight2', 0.5, 2.5),
        # 特别生肖
        'four_omit_boost': trial.suggest_float('four_omit_boost', 1.0, 5.0),
        # 特别号规则部分
        'cold_threshold': trial.suggest_int('cold_threshold', 6, 18),
        'neighbor1': trial.suggest_float('neighbor1', 2.0, 7.0),
        'neighbor2': trial.suggest_float('neighbor2', 0.5, 4.0),
        'recent_penalty': trial.suggest_float('recent_penalty', 0.5, 0.98),
        # 特别号融合权重
        'lgb_weight': trial.suggest_float('lgb_weight', 0.3, 0.8),
    }
    return evaluate_all(all_issues, params)

# ---------- 主程序 ----------

def main():
    parser = argparse.ArgumentParser(description="全自动超参数优化 v2（近期120期 + 混合融合）")
    parser.add_argument('--db', default='newmacau_marksix.db', help='数据库路径')
    parser.add_argument('--trials', type=int, default=300, help='Optuna 试验次数')
    args = parser.parse_args()

    # 加载最近120期数据
    conn = connect_db(args.db)
    try:
        all_issues = load_all_issues(conn, recent=120)
        if len(all_issues) < 60:
            print(f"错误：最近120期数据不足（只有{len(all_issues)}期），请先同步数据。")
            sys.exit(1)
        print(f"使用最近120期数据，实际有效回测期数：{len(all_issues)-40} 期")
    finally:
        conn.close()

    # 创建 study
    study = optuna.create_study(
        direction='maximize',
        study_name='optimize_v2_120',
        storage='sqlite:///optuna_results_v2_120.db',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
    )
    study.optimize(lambda trial: objective(trial, all_issues), n_trials=args.trials, show_progress_bar=True)

    # 输出结果
    print("\n" + "="*60)
    print("优化完成！最佳参数组合（基于最近120期）：")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"最佳加权得分: {study.best_value:.4f}")
    print("="*60)

    # 保存到 JSON
    best = study.best_params
    best["score"] = study.best_value
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print("最佳参数已保存至 best_params.json")

if __name__ == '__main__':
    main()
