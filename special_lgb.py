import numpy as np
import lightgbm as lgb
import json, sqlite3
from collections import Counter

def build_special_features(conn, issue_no, candidate_num):
    # 特征构建（严格使用 <= 但期数据）
    # 返回：遗漏值、遗漏二阶差、特别号生肖转移概率、尾数转移概率、
    #      与主号池重叠度、最近特别号邻号加分等
    features = []
    # ... 实现可参考原 _generate_special_number_v4 中的计算，但转换成数值数组
    return features

def train_special_classifier(conn, recent_issues=300):
    X, y = [], []
    draws = conn.execute(
        "SELECT issue_no, special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
    for i in range(50, len(draws)):
        target_sp = draws[i]["special_number"]
        # 对当期所有候选号码（例如排除主号池后）构建特征，标签为是否为实际特别号
        candidates = list(range(1,50))  # 简单所有号码
        for n in candidates:
            feat = build_special_features(conn, draws[i]["issue_no"], n)
            X.append(feat)
            y.append(1 if n == target_sp else 0)
    model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, class_weight='balanced')
    model.fit(np.array(X), np.array(y))
    return model

def predict_special_proba(model, conn, issue_no, candidates):
    probs = []
    for n in candidates:
        feat = build_special_features(conn, issue_no, n)
        proba = model.predict_proba([feat])[0, 1]
        probs.append((n, proba))
    return sorted(probs, key=lambda x: -x[1])
