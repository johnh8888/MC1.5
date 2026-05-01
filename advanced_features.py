#!/usr/bin/env python3
""" 高级特征工程：为生肖和号码预测提供深层信号 """
from collections import Counter
from typing import List, Dict

def get_zodiac_momentum(history_zodiacs: List[str], window: int = 10) -> Dict[str, float]:
    """
    生肖动量：最近 window 期内，每个生肖出现次数的变化率。
    正值表示变热，负值表示变冷。
    """
    if len(history_zodiacs) < window * 2:
        return {}
    recent = history_zodiacs[:window]
    older = history_zodiacs[window:window*2]
    recent_cnt = Counter(recent)
    older_cnt = Counter(older)
    momentum = {}
    for z in set(list(recent_cnt.keys()) + list(older_cnt.keys())):
        momentum[z] = recent_cnt.get(z, 0) - older_cnt.get(z, 0)
    return momentum

def get_zodiac_cycle_position(history_zodiacs: List[str], target_z: str) -> float:
    """
    生肖周期位置：目标生肖距离上次出现已经过了多少期。
    同时返回该生肖在历史中的平均回补周期。
    """
    distances = []
    last_seen = -1
    for i, z in enumerate(history_zodiacs):
        if z == target_z:
            if last_seen != -1:
                distances.append(i - last_seen)
            last_seen = i
    if not distances:
        return len(history_zodiacs)  # 从未出现过
    avg_cycle = sum(distances) / len(distances)
    # 当前距离上一次出现
    current_gap = 0
    for z in history_zodiacs:
        if z == target_z:
            break
        current_gap += 1
    if avg_cycle == 0:
        return float(current_gap)
    return current_gap / avg_cycle   # 相对周期位置

def get_hot_zodiac_clusters(history_zodiacs: List[str], window: int = 10) -> List[str]:
    """
    热号聚类：最近 window 期出现最频繁的生肖组。
    """
    recent = history_zodiacs[:window]
    cnt = Counter(recent)
    avg = sum(cnt.values()) / len(cnt) if cnt else 0
    return [z for z, c in cnt.items() if c >= avg * 1.5]

def get_zodiac_pair_scores(history_zodiacs: List[str], window: int = 20) -> Dict[str, float]:
    """
    生肖配对得分：同时出现的生肖组合频率。
    """
    pairs = Counter()
    for i in range(len(history_zodiacs) - 1):
        pair = tuple(sorted(history_zodiacs[i:i+2]))
        pairs[pair] += 1
    scores = {}
    for (z1, z2), cnt in pairs.items():
        scores[f"{z1}+{z2}"] = cnt / window
    return scores
