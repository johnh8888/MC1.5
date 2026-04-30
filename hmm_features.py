import numpy as np
from hmmlearn import hmm

def train_hmm_per_number(conn):
    # 对每个号码，根据历史出现序列训练一个 GaussianHMM（用遗漏值作为观测）
    models = {}
    for num in range(1, 50):
        series = []
        # 读取历史遗漏变化...
        # 训练HMM
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        model.fit(series)
        models[num] = model
    return models

def get_hmm_state_probas(models, conn, num):
    # 返回 [p_cold, p_warm, p_hot]
    ...
