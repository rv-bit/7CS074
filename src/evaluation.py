import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
    accuracy_score
)
from sklearn.model_selection import KFold, cross_val_score

import numpy as np

    
def regression_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics"""
    return {
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def classification_metrics(y_true, y_pred, average='weighted') -> dict:
    """Calculate classification metrics"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def get_classification_report(y_true, y_pred, target_names=None):
    """Get detailed classification report"""
    return classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        zero_division=0
    )

def calculate_silhouette_score(X, labels):
    """Calculate silhouette score for clustering"""
    mask = labels != -1
    unique_labels = len(set(labels[mask]))
    if unique_labels < 2:  # Need at least 2 clusters
        return None
    
    try:
        return silhouette_score(X[mask], labels[mask])
    except:
        return None
    
def interpret_rules(rules_df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Print human-readable interpretation of top association rules.
    
    Args:
        rules_df: DataFrame with association rules
        top_n: Number of top rules to display
    """
    if rules_df.empty:
        print("No rules to interpret")
        return
    
    print(f"\n{'='*80}")
    print(f"TOP {top_n} ASSOCIATION RULES")
    print(f"{'='*80}\n")
    
    for idx, row in rules_df.head(top_n).iterrows():
        antecedents = ', '.join([item.replace(':', ' = ') for item in row['antecedents']])
        consequents = ', '.join([item.replace(':', ' = ') for item in row['consequents']])
        
        print(f"Rule {idx + 1}:")
        print(f"  IF {antecedents}")
        print(f"  THEN {consequents}")
        print(f"  Support: {row['support']:.3f} | Confidence: {row['confidence']:.3f} | Lift: {row['lift']:.3f}")
        print(f"  → This means: {row['confidence']*100:.1f}% of cars with {antecedents}")
        print(f"    also have {consequents}")
        print(f"    (This is {row['lift']:.2f}x more likely than random)\n")

def get_rules_summary(rules_df: pd.DataFrame) -> dict:
    """
    Get summary statistics of association rules.
    
    Args:
        rules_df: DataFrame with association rules
    
    Returns:
        Dictionary with summary statistics
    """
    if rules_df.empty:
        return {
            'total_rules': 0,
            'avg_support': 0,
            'avg_confidence': 0,
            'avg_lift': 0
        }
    
    return {
        'total_rules': len(rules_df),
        'avg_support': rules_df['support'].mean(),
        'avg_confidence': rules_df['confidence'].mean(),
        'avg_lift': rules_df['lift'].mean(),
        'max_lift': rules_df['lift'].max(),
        'min_support': rules_df['support'].min(),
        'max_support': rules_df['support'].max()
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