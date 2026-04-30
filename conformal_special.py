import numpy as np
from mapie.classification import MapieClassifier

def train_conformal_special(model, X_calib, y_calib):
    mapie = MapieClassifier(model, cv="prefit", method="score")
    mapie.fit(X_calib, y_calib)
    return mapie

def predict_with_confidence(mapie, X, alpha=0.1):
    y_pred, y_set = mapie.predict(X, alpha=alpha)
    return y_set  # 包含预测类别的集合
