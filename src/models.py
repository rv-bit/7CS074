from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from src.evaluation import regression_metrics

def get_regression_model(model_name: str):
    if model_name == "linear":
        return LinearRegression()
    elif model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
    else:
        raise ValueError("Unknown model type")

def get_classification_model(model_name: str):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
    )
    else:
        raise ValueError("Unknown model type")
    

def get_clustering_model(k: int = 3):
    return KMeans(n_clusters=k, random_state=42)


def train_model(X, y, model_type="random_forest"):
    model = get_regression_model(model_type)
    model.fit(X, y)
    return model


def automatic_make_model_selection(
    df,
    feature_cols,
    target_col,
    model_type="random_forest",
    min_samples_per_make=200
):
    all_models = {}
    trained_models_per_make = {}

    # Global model
    X_global = df[feature_cols]
    y_global = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X_global, y_global, test_size=0.2, random_state=42
    )

    global_model = train_model(X_train, y_train, model_type=model_type)
    global_predictions = global_model.predict(X_test)
    global_metrics = regression_metrics(y_test, global_predictions)

    all_models['GLOBAL'] = {
        'model': global_model,
        'metrics': global_metrics
    }

    # Per-make models
    for make, group in df.groupby("make"):
        if len(group) < min_samples_per_make:
            continue

        X = group[feature_cols]
        y = group[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = regression_metrics(y_test, predictions)

        all_models[make] = {
            'model': model,
            'metrics': metrics
        }
        
        trained_models_per_make[make] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'metrics': metrics,
            'predictions': predictions
        }

    return all_models, trained_models_per_make