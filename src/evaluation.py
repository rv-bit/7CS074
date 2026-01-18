from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import KFold, cross_val_score

import numpy as np

def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAPE":  np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def select_best_model_cv(
    X,
    y,
    models: dict,
    cv_splits: int = 5,
    scoring: str = "neg_mean_absolute_error"
):
    """
        Selects the best model using cross-validation.
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    scores = {}
    for name, model in models.items():
        cv_score = cross_val_score(
            clone(model),
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        ).mean()
        scores[name] = cv_score

    best_model_name = max(scores, key=scores.get)
    best_model = clone(models[best_model_name])
    best_model.fit(X, y)

    return best_model_name, best_model, scores