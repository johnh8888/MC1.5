# adaptive_features.py
import json
from collections import Counter
ALL_NUMBERS = list(range(1, 50))

def calc_span_and_sum(numbers):
    if not numbers or len(numbers) < 6: return 0, 0
    sorted_nums = sorted(numbers)
    return sorted_nums[-1] - sorted_nums[0], sum(numbers)

def get_last_issue_context(conn):
    """
    获取上一期开奖的"大势"特征，作为下一期的参考。
    返回一个 dict，包含：跨度、和值、奇偶比、大小比、冷热号分布等。
    """
    last = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT 1"
    ).fetchone()
    if not last: return {}

    nums = json.loads(last['numbers_json'])
    special = last['special_number']
    span, total_sum = calc_span_and_sum(nums)

    # 奇偶比 (0-1范围)
    odd_cnt = sum(1 for n in nums if n % 2 == 1)
    odd_ratio = odd_cnt / 6.0

    # 大小比 (以25为界)
    big_cnt = sum(1 for n in nums if n > 25)
    big_ratio = big_cnt / 6.0

    # 冷热号分布：最近30期内各号码出现次数
    # (简化版：只返回平均遗漏值)
    recent_30 = conn.execute(
        "SELECT numbers_json FROM draws ORDER BY draw_date DESC LIMIT 30"
    ).fetchall()
    freq_30 = Counter()
    for r in recent_30:
        freq_30.update(json.loads(r['numbers_json']))
    mean_freq = sum(freq_30.values()) / len(freq_30)
    hot_cnt = sum(1 for n in nums if freq_30.get(n, 0) > mean_freq)
    cold_cnt = 6 - hot_cnt

    return {
        'last_span': span,
        'last_sum': total_sum,
        'last_odd_ratio': odd_ratio,
        'last_big_ratio': big_ratio,
        'last_hot_count': hot_cnt,
        'last_cold_count': cold_cnt,
        'last_special': special
    }
