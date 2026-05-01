#!/usr/bin/env python3
""" LSTM 模型：支持可变序列长度 """
import sqlite3, json, argparse
import numpy as np
from pathlib import Path

ZODIAC_MAP = { ... }  # 与主脚本一致，此处省略重复
ZODIAC_LIST = list(ZODIAC_MAP.keys())

def get_zodiac_by_number(n): ...

def connect_db(path): ...

def build_sequence_data(conn, seq_len=30):
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date ASC"
    ).fetchall()
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
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("[LSTM] 未安装 TensorFlow，跳过训练。")
        return

    X, y = build_sequence_data(conn, seq_len)
    if len(X) < 100:
        print(f"[LSTM] 数据不足（{len(X)} 条），跳过训练")
        return

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
              epochs=epochs, batch_size=32, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=1)
    model.save(model_path)
    print(f"[LSTM] 模型已保存至 {model_path}")

def predict_lstm_proba(conn, model_path='lstm_zodiac.h5', seq_len=30):
    if not Path(model_path).exists():
        return None
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None
    model = load_model(model_path)
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC LIMIT ?",
        (seq_len,)
    ).fetchall()
    if len(rows) < seq_len:
        return None
    features = []
    for row in rows[::-1]:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='newmacau_marksix.db')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=30)  # 减少点时间
    args = parser.parse_args()
    conn = connect_db(args.db)
    train_lstm(conn, seq_len=args.seq_len, epochs=args.epochs)
    conn.close()
