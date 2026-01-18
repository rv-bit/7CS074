import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_price_vs_mpg(df: pd.DataFrame):
    sns.scatterplot(data=df, x='mpg', y='price')
    plt.title("Price vs MPG")
    plt.show()

def plot_actual_vs_predicted(actuals, predictions, model_name, R_SQUARE):
    plt.figure(figsize=(10,8))
    plt.scatter(actuals, predictions, alpha=0.4, c="teal", label="Predicted vs Actual", s=50)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)],
                "r--", label="Perfect Prediction (Actual = Predicted)", linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Scatter Plot of Actual vs Predicted Price\nR² = {R_SQUARE:.3f} - {model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_feature_importances(values, feature_names, title):
    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": values
        })
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_clusters(df: pd.DataFrame, cluster_col: str):
    sns.scatterplot(
        data=df,
        x='mpg',
        y='price',
        hue=cluster_col,
        palette='Set2'
    )
    plt.title("Car Clusters")
    plt.show()