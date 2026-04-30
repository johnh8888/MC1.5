#!/usr/bin/env python3
"""
全自动超参数优化脚本（生肖 + 特别号）
数据源：在线 API（由 sync 命令拉取）
使用方法：
    python hyper_optimize.py --db newmacau_marksix.db --trials 300
"""

import sqlite3
import json
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
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


def load_all_issues(conn) -> List[Tuple[str, List[int], int]]:
    """返回所有期号的历史数据：[(issue_no, [n1..n6], special), ...] 按日期升序"""
    rows = conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
    data = []
    for r in rows:
        data.append((r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])))
    return data


# ---------- 预测函数（参数可调） ----------

def predict_single_zodiac(history: list, window: int, recency_w: float) -> str:
    scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = history[-window:] if len(history) >= window else history
    for idx, (_, nums, sp) in enumerate(recent[::-1]):
        w = recency_w / (1.0 + idx * 0.15)
        for n in nums:
            scores[get_zodiac(n)] += w
        scores[get_zodiac(sp)] += w * 2.0
    return max(scores, key=scores.get)


def predict_two_zodiac(history: list, hot_w: float, cold_w: float) -> List[str]:
    specials = [sp for _, _, sp in history[-10:]]
    hot_counter = Counter([get_zodiac(sp) for sp in specials])
    hot = max(hot_counter, key=hot_counter.get)

    omission = {z: 0 for z in ZODIAC_MAP}
    for i in range(len(history)):
        _, nums, sp = history[-(i + 1)]
        for z in ZODIAC_MAP:
            if omission[z] == 0:
                omission[z] = i + 1
        for n in nums:
            omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0

    cold = max((z for z in ZODIAC_MAP if z != hot), key=lambda z: omission[z])
    return [hot, cold]


def predict_three_zodiac(history: list, min_omit: int) -> List[str]:
    two = predict_two_zodiac(history, 1.0, 1.0)
    omission = {z: 0 for z in ZODIAC_MAP}
    for i in range(len(history)):
        _, nums, sp = history[-(i + 1)]
        for z in ZODIAC_MAP:
            if omission[z] == 0:
                omission[z] = i + 1
        for n in nums:
            omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    extra = [z for z, o in omission.items() if o >= min_omit and z not in two]
    if extra:
        third = max(extra, key=lambda z: omission[z])
    else:
        third = max((z for z in ZODIAC_MAP if z not in two), key=lambda z: omission[z])
    return two[:2] + [third]


def predict_four_zodiac(history: list, omit_boost: float) -> List[str]:
    omission = {z: 0 for z in ZODIAC_MAP}
    for i in range(len(history)):
        _, nums, sp = history[-(i + 1)]
        for z in ZODIAC_MAP:
            if omission[z] == 0:
                omission[z] = i + 1
        for n in nums:
            omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    sorted_cold = sorted(omission.items(), key=lambda x: (-x[1], x[0]))
    picks = [z for z, _ in sorted_cold[:4]]
    latest_z = get_zodiac(history[-1][2])
    if latest_z not in picks:
        picks[-1] = latest_z
    return picks[:4]


def predict_special_numbers(history: list, cold_threshold: int,
                            neighbor_score1: float, neighbor_score2: float,
                            recent_penalty: float) -> List[int]:
    specials = [sp for _, _, sp in history]
    if len(specials) < 12:
        return [1, 2, 3]
    omission = {}
    for i, sp in enumerate(specials):
        omission[sp] = omission.get(sp, i + 1) if sp not in omission else min(omission[sp], i + 1)
    cold = sorted(
        [n for n in ALL_NUMBERS if omission.get(n, 999) >= cold_threshold],
        key=lambda n: omission.get(n, 999), reverse=True
    )[:2]
    if len(cold) < 2:
        extra = sorted([n for n in ALL_NUMBERS if n not in cold], key=lambda n: omission.get(n, 999), reverse=True)
        while len(cold) < 2 and extra:
            cold.append(extra.pop(0))
    picks = cold[:2]
    latest = specials[-1]
    neighbors = [n for n in ALL_NUMBERS if abs(n - latest) == 1 and n not in picks]
    if neighbors and len(picks) < 3:
        picks.append(max(neighbors, key=lambda n: omission.get(n, 999)))
    elif len(picks) < 3:
        rest = sorted([n for n in ALL_NUMBERS if n not in picks], key=lambda n: omission.get(n, 999), reverse=True)
        while len(picks) < 3 and rest:
            picks.append(rest.pop(0))
    return picks[:3]


# ---------- 综合评估 ----------

def evaluate_all(history_issues: list, params: dict) -> float:
    single_hits = two_hits = three_hits = four_hits = special_hits = 0
    total = 0
    min_len = 40
    for i in range(min_len, len(history_issues)):
        past = history_issues[:i]
        cur_nums, cur_sp = history_issues[i][1], history_issues[i][2]
        cur_zodiacs = set(get_zodiac(n) for n in cur_nums)
        cur_zodiacs.add(get_zodiac(cur_sp))

        # 一生肖
        single = predict_single_zodiac(past, params['single_window'], params['single_recency_w'])
        if single in cur_zodiacs:
            single_hits += 1

        # 二生肖
        two = predict_two_zodiac(past, params['two_hot_w'], params['two_cold_w'])
        if any(z in cur_zodiacs for z in two):
            two_hits += 1

        # 三生肖（至少中2只）
        three = predict_three_zodiac(past, params['three_min_omit'])
        if sum(1 for z in three if z in cur_zodiacs) >= 2:
            three_hits += 1

        # 特别生肖（中1只）
        four = predict_four_zodiac(past, params['four_omit_boost'])
        if any(z in cur_zodiacs for z in four):
            four_hits += 1

        # 特别号精选
        sp_pred = predict_special_numbers(past, params['cold_threshold'],
                                          params['neighbor1'], params['neighbor2'],
                                          params['recent_penalty'])
        if cur_sp in sp_pred:
            special_hits += 1

        total += 1

    if total == 0:
        return 0.0
    return (single_hits/total + two_hits/total + three_hits/total +
            four_hits/total + special_hits/total) / 5


# ---------- Optuna 目标函数 ----------

def objective(trial: optuna.Trial, all_issues: list) -> float:
    params = {
        # 一生肖
        'single_window': trial.suggest_int('single_window', 6, 20),
        'single_recency_w': trial.suggest_float('single_recency_w', 0.8, 2.0),
        # 二生肖
        'two_hot_w': trial.suggest_float('two_hot_w', 0.8, 2.5),
        'two_cold_w': trial.suggest_float('two_cold_w', 0.8, 2.5),
        # 三生肖
        'three_min_omit': trial.suggest_int('three_min_omit', 3, 8),
        # 特别生肖
        'four_omit_boost': trial.suggest_float('four_omit_boost', 1.0, 4.0),
        # 特别号
        'cold_threshold': trial.suggest_int('cold_threshold', 8, 18),
        'neighbor1': trial.suggest_float('neighbor1', 2.0, 7.0),
        'neighbor2': trial.suggest_float('neighbor2', 1.0, 4.0),
        'recent_penalty': trial.suggest_float('recent_penalty', 0.65, 0.95),
    }
    return evaluate_all(all_issues, params)


# ---------- 主程序 ----------

def main():
    parser = argparse.ArgumentParser(description="全自动超参数优化")
    parser.add_argument('--db', default='newmacau_marksix.db', help='数据库路径')
    parser.add_argument('--trials', type=int, default=300, help='Optuna 试验次数')
    args = parser.parse_args()

    conn = connect_db(args.db)
    try:
        all_issues = load_all_issues(conn)
        if len(all_issues) < 60:
            print("数据不足，至少需要60期历史。请先同步数据！")
            sys.exit(1)
    finally:
        conn.close()

    # 创建 study，结果保存到数据库以便断点续传
    study = optuna.create_study(
        direction='maximize',
        study_name='full_optimize',
        storage='sqlite:///optuna_results.db',
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, all_issues), n_trials=args.trials, show_progress_bar=True)

    # 输出并保存最佳参数
    print("\n" + "=" * 60)
    print("优化完成！最佳参数组合：")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"最佳综合得分 (平均命中率): {study.best_value:.4f}")
    print("=" * 60)

    best = study.best_params
    best["score"] = study.best_value
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print("最佳参数已保存至 best_params.json")


if __name__ == '__main__':
    main()
