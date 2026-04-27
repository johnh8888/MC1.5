#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xgboost as xgb

ALL_NUMBERS = list(range(1, 50))
FEATURE_WINDOW_DEFAULT = 10

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH_DEFAULT = SCRIPT_DIR / "xgb_ensemble_model.pkl"
STRATEGIES = ["hot_v1", "cold_rebound_v1", "momentum_v1", "balanced_v1", "pattern_mined_v1"]

GenerateStrategyFn = Callable[..., Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]]
LoadRecentDrawsFn = Callable[[sqlite3.Connection, int], List[List[int]]]


def _omission_map(draws: Sequence[Sequence[int]]) -> Dict[int, float]:
    """Return first-seen omission distance for each number."""
    omission = {n: float(len(draws) + 1) for n in ALL_NUMBERS}
    for idx, draw in enumerate(draws):
        seen = set()
        for raw in draw:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 49:
                seen.add(n)
        for n in seen:
            if omission[n] > float(idx + 1):
                omission[n] = float(idx + 1)
    return omission


def compute_market_temperature(draws: Sequence[Sequence[int]]) -> Dict[str, float]:
    if not draws:
        return {"cold_ratio": 0.0, "zone_entropy": 0.0}

    counts = {n: 0 for n in ALL_NUMBERS}
    zone_counts = [0.0] * 5
    for draw in draws:
        for raw in draw:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= 49:
                counts[n] += 1
                zone_counts[min(4, (n - 1) // 10)] += 1.0

    cold_ratio = sum(1 for n in ALL_NUMBERS if counts[n] == 0) / float(len(ALL_NUMBERS))

    zone_total = float(sum(zone_counts))
    if zone_total <= 0:
        return {"cold_ratio": float(cold_ratio), "zone_entropy": 0.0}

    probs = [c / zone_total for c in zone_counts if c > 0]
    if not probs:
        zone_entropy = 0.0
    else:
        zone_entropy = 0.0
        for p in probs:
            zone_entropy -= p * float(np.log(p + 1e-12))
        zone_entropy = zone_entropy / float(np.log(5.0)) if zone_entropy > 0 else 0.0

    return {"cold_ratio": float(cold_ratio), "zone_entropy": float(zone_entropy)}


def _score_vector_to_probabilities(score_map: Dict[int, float]) -> np.ndarray:
    values = np.asarray([float(score_map.get(n, 0.0)) for n in ALL_NUMBERS], dtype=np.float32)
    values = values - float(values.max())
    exp = np.exp(values)
    denom = float(exp.sum()) or 1.0
    return exp / denom


def _build_feature_vector(
    draws: Sequence[Sequence[int]],
    temperature: Dict[str, float],
    generate_strategy_fn: Optional[GenerateStrategyFn] = None,
) -> np.ndarray:
    recent = list(draws[:FEATURE_WINDOW_DEFAULT])
    temp = compute_market_temperature(recent)
    omission = _omission_map(recent)
    features: List[float] = [
        float(len(recent)),
        float(temperature.get("cold_ratio", temp["cold_ratio"])),
        float(temperature.get("zone_entropy", temp["zone_entropy"])),
    ]

    # number-level features for all 49 numbers: omission, tail, zodiac, normalized id
    for n in ALL_NUMBERS:
        features.append(float(omission.get(n, 0.0)))
        features.append(float(n % 10))
        features.append(float((n - 1) // 12))
        features.append(float(n) / 49.0)

    if generate_strategy_fn is None:
        for _ in STRATEGIES:
            features.extend([0.0] * 49)
            features.extend([0.0, 0.0, 0.0])
        return np.asarray(features, dtype=np.float32)

    for strategy in STRATEGIES:
        try:
            picks, special_number, special_score, score_map = generate_strategy_fn(
                recent,
                strategy,
                conn=None,
                issue_no=None,
            )
        except Exception:
            picks = []
            special_number = 0
            special_score = 0.0
            score_map = {n: 0.0 for n in ALL_NUMBERS}

        probs = _score_vector_to_probabilities(score_map)
        features.extend(probs.tolist())
        features.append(float(special_number or 0))
        features.append(float(special_score or 0.0))
        features.append(float(len(picks)))

    return np.asarray(features, dtype=np.float32)


def _extract_training_rows(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC"
    ).fetchall()


def train_ensemble_model(
    conn: sqlite3.Connection,
    generate_strategy_fn: GenerateStrategyFn,
    load_recent_draws_fn: LoadRecentDrawsFn,
    model_path: Path = MODEL_PATH_DEFAULT,
) -> xgb.XGBClassifier:
    rows = _extract_training_rows(conn)
    if len(rows) < 20:
        raise RuntimeError("Not enough historical draws to train ensemble model.")

    parsed_draws = [json.loads(r["numbers_json"]) for r in rows]
    X: List[np.ndarray] = []
    y: List[int] = []

    for i in range(FEATURE_WINDOW_DEFAULT, len(parsed_draws)):
        history = list(reversed(parsed_draws[max(0, i - FEATURE_WINDOW_DEFAULT):i]))
        temperature = compute_market_temperature(history)

        next_draw = {int(n) for n in parsed_draws[i] if 1 <= int(n) <= 49}
        score_features = _build_feature_vector(history, temperature, generate_strategy_fn)
        for n in ALL_NUMBERS:
            row_feat = score_features.copy()
            row_feat = np.concatenate([
                row_feat,
                np.asarray([
                    float(n) / 49.0,
                    float(n % 10),
                    float((n - 1) // 12),
                ], dtype=np.float32),
            ])
            X.append(row_feat)
            y.append(1 if n in next_draw else 0)

    if not X:
        raise RuntimeError("Failed to build training dataset for ensemble model.")

    X_arr = np.vstack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int32)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=6,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_arr, y_arr)

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    return model


def load_or_train_ensemble_model(
    conn: sqlite3.Connection,
    generate_strategy_fn: GenerateStrategyFn,
    load_recent_draws_fn: LoadRecentDrawsFn,
    model_path: Path = MODEL_PATH_DEFAULT,
    force_retrain: bool = False,
) -> xgb.XGBClassifier:
    if force_retrain and model_path.exists():
        model_path.unlink()
    if model_path.exists():
        with model_path.open("rb") as f:
            return pickle.load(f)
    return train_ensemble_model(
        conn,
        generate_strategy_fn=generate_strategy_fn,
        load_recent_draws_fn=load_recent_draws_fn,
        model_path=model_path,
    )


def ensemble_predict(
    scores_dict: Dict[str, Dict[int, float]],
    temperature: Dict[str, float],
    model: xgb.XGBClassifier,
) -> Dict[int, float]:
    strategy_names = STRATEGIES
    combined_scores = {n: 0.0 for n in ALL_NUMBERS}

    for strategy in strategy_names:
        score_map = scores_dict.get(strategy, {})
        for n in ALL_NUMBERS:
            combined_scores[n] += float(score_map.get(n, 0.0))

    max_score = max(combined_scores.values()) if combined_scores else 0.0
    min_score = min(combined_scores.values()) if combined_scores else 0.0
    if max_score != min_score:
        combined_scores = {n: (v - min_score) / (max_score - min_score) for n, v in combined_scores.items()}

    base_features = _build_feature_vector([[]], temperature, generate_strategy_fn=None)
    probs: Dict[int, float] = {}

    for n in ALL_NUMBERS:
        feature_vector = np.concatenate([base_features, np.asarray([float(n) / 49.0], dtype=np.float32)])
        try:
            pred = float(model.predict_proba(feature_vector.reshape(1, -1))[0, 1])
        except Exception:
            pred = 0.0
        probs[n] = float(0.7 * pred + 0.3 * combined_scores.get(n, 0.0))

    total = sum(probs.values()) or 1.0
    return {n: v / total for n, v in probs.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost ensemble trainer for New Macau Mark Six")
    parser.add_argument("--db", default=str(SCRIPT_DIR / "newmacau_marksix.db"), help="SQLite database path")
    parser.add_argument("--train", action="store_true", help="Train or retrain the ensemble model")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        print("This module is intended to be imported by newmacau_marksix.py.")
        if args.train:
            raise SystemExit("Training from CLI requires passing generate_strategy_fn and load_recent_draws_fn from the main script.")
        else:
            raise SystemExit("Loading from CLI requires passing generate_strategy_fn and load_recent_draws_fn from the main script.")
    finally:
        conn.close()
