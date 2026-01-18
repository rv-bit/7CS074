from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

import src.evaluation as evaluation
import src.features as features

import global_vars

def get_regression_candidate_models(random_state=42):
    """
        Returns a list of different regression models, starting with a base model, and hyper tunning others, this can allow to find which is the best model in the case of regression for our current dataset of used cars
    """
    return {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=5
        )
    }

def get_classification_candidate_models(random_state=42):
    """
        Returns a list of different classification models, starting with a base model, and hyper tunning others, this can allow to find which is the best model in the case of classification for our current dataset of used cars
    """
    return {
        "random_forrest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state
        )
    }

def get_clustering_candidate_models(clusters=3, random_state=42):
    """
        Returns a list of different clustering models, starting with a base model, and hyper tunning others, this can allow to find which is the best model in the case of clustering for our current dataset of used cars
    """
    return {
        'k-means': KMeans(n_clusters=clusters, random_state=random_state)
    }

def get_type_models(type_models):
    """
        Returns a the candidate models base on the argument passed, if not found raise error
    """
    match type_models:
        case 'regression':
            return get_regression_candidate_models()
        case 'classification':
            return get_classification_candidate_models()
        case 'clustering':
            return get_clustering_candidate_models(),

        case _:
            raise ValueError("Type model isn't found in the list")

def automatic_make_model_selection(
    df,
    target_col,

    type_models='regression',
    
    # Tunning
    min_samples_per_make=200,
    cv_splits=5
):
    candidate_models = get_type_models(type_models)
    
    all_models = {}
    trained_models_per_make = {}

    X_global, global_encoder = features.engineer_features(
        df,
        global_vars.CATEGORICAL_FEATURES_GLOBAL
    )
    Y_global = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X_global, Y_global, test_size=0.2, random_state=42
    )

    best_name, best_model, cv_scores = evaluation.select_best_model_cv(
        X_train, y_train, candidate_models, cv_splits
    )
    
    global_predictions = best_model.predict(X_test)
    global_metrics = evaluation.regression_metrics(y_test, global_predictions)

    all_models["GLOBAL"] = {
        "best_model": best_name,
        "model": best_model,
        "cv_scores": cv_scores,
        "metrics": global_metrics
    }

    # Per-make models
    # If the length of each dataframe of group is lower than the minimum sample variable, we shall pass and not create a 'per-make' model
    # This is to make sure that algorithms like 'Random Forrest' gets trained on larger sets of data, as indented, if the condition is true, the make would fallback into the global model above.
    for make, group_df in df.groupby("make"):
        if len(group_df) < min_samples_per_make:
            continue
        
        X, encoder = features.engineer_features(
            group_df, 
            global_vars.CATEGORICAL_FEATURES_PER_MAKE
        )
        Y = group_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        best_name, best_model, cv_scores = evaluation.select_best_model_cv(
            X_train, y_train, candidate_models, cv_splits
        )

        predictions = best_model.predict(X_test)
        metrics = evaluation.regression_metrics(y_test, predictions)

        all_models[make] = {
            "best_model": best_name,
            "model": best_model,
            "cv_scores": cv_scores,
            "metrics": metrics
        }

        trained_models_per_make[make] = {
            "model": best_model,
            "encoder": encoder,
            "feature_names": X_train.columns,
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions,
            "metrics": metrics
        }

    return all_models, trained_models_per_make