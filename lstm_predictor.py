#!/usr/bin/env python3
""" LSTM 序列模型：预测下一期各生肖的出现概率 """
import sqlite3, json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

ZODIAC_MAP = { ... }  # 与主脚本一致，省略
ZODIAC_LIST = list(ZODIAC_MAP.keys())  # 12生肖顺序固定

def connect_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def build_sequence_data(conn, seq_len=30):
    """
    构建训练/预测所需的时序特征
    返回: X (样本数, seq_len, 特征维度), y (样本数, 12) 
          对于每一期，X是前seq_len期的特征，y是当期各生肖是否出现（主号+特别号）
    """
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
    # 每期提取12维向量：每个生肖是否出现（1/0）
    all_features = []
    for row in rows:
        nums = json.loads(row["numbers_json"])
        sp = int(row["special_number"])
        vec = np.zeros(12)
        for n in nums:
            z = get_zodiac_by_number(n)
            vec[ZODIAC_LIST.index(z)] = 1.0
        z_sp = get_zodiac_by_number(sp)
        vec[ZODIAC_LIST.index(z_sp)] = 1.0
        all_features.append(vec)

    X, y = [], []
    for i in range(seq_len, len(all_features)):
        X.append(all_features[i-seq_len:i])
        y.append(all_features[i])
    return np.array(X), np.array(y)

def train_lstm(conn, model_path='lstm_zodiac.h5', seq_len=30, epochs=50):
    X, y = build_sequence_data(conn, seq_len)
    if len(X) < 100:
        print("[LSTM] 数据不足，跳过训练")
        return

    # 划分训练/验证集（时序，不可随机打乱）
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(64, input_shape=(seq_len, 12), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(12, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=32, callbacks=[EarlyStopping(patience=10)],
              verbose=1)
    model.save(model_path)
    print(f"[LSTM] 模型已保存至 {model_path}")

def predict_lstm_proba(conn, model_path='lstm_zodiac.h5', seq_len=30):
    """ 返回对下一期12生肖的概率向量 """
    if not Path(model_path).exists():
        return None
    model = load_model(model_path)
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT ?",
        (seq_len,)
    ).fetchall()
    if len(rows) < seq_len:
        return None
    # 构建最近seq_len期的特征
    features = []
    for row in rows[::-1]:  # 从旧到新
        nums = json.loads(row["numbers_json"])
        sp = int(row["special_number"])
        vec = np.zeros(12)
        for n in nums:
            z = get_zodiac_by_number(n)
            vec[ZODIAC_LIST.index(z)] = 1.0
        z_sp = get_zodiac_by_number(sp)
        vec[ZODIAC_LIST.index(z_sp)] = 1.0
        features.append(vec)
    X_input = np.array(features[-seq_len:]).reshape(1, seq_len, 12)
    proba = model.predict(X_input, verbose=0)[0]
    return dict(zip(ZODIAC_LIST, proba))
