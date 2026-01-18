from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

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
        ),
        "decision_tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
    }

def get_clustering_candidate_models(clusters=3, random_state=42):
    """
        Returns a list of different clustering models, starting with a base model, and hyper tunning others, this can allow to find which is the best model in the case of clustering for our current dataset of used cars
    """
    return {
        'k-means': KMeans(n_clusters=clusters, random_state=random_state)
    }

def get_type_models(type_models, **kwargs):
    """
    Returns the candidate models based on the argument passed, if not found raise error
    """
    match type_models:
        case 'regression':
            return get_regression_candidate_models(kwargs.get('random_state', 42))
        case 'classification':
            return get_classification_candidate_models(kwargs.get('random_state', 42))
        case 'clustering':
            return get_clustering_candidate_models(
                kwargs.get('clusters', 4), 
                kwargs.get('random_state', 42)
            )

        case _:
            raise ValueError("Type model isn't found in the list")