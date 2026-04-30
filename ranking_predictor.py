import lightgbm as lgb
import numpy as np

def build_ranking_features(conn, issue_no):
    # 为1-49每个号码构建特征矩阵 (49, n_features)
    ...
    return np.array(features)

def train_ranking_model(conn, recent=500):
    X, y, groups = [], [], []
    # 构造排序样本：每期为一个组(group)，label为该号码是否在开奖中(1/0)
    # 使用 lgb.Dataset 设置 group 参数
    model = lgb.LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        n_estimators=300,
        max_depth=8,
        importance_type="gain"
    )
    model.fit(X, y, group=groups)
    return model

def predict_ranking(model, conn, issue_no):
    features = build_ranking_features(conn, issue_no)
    scores = model.predict(features)
    top6 = np.argsort(scores)[-6:][::-1] + 1
    return list(top6)
