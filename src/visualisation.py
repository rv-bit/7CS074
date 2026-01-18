import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Set default styling style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Regression

def plot_price_vs_mpg(df: pd.DataFrame):
    """Scatter plot of price vs MPG"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mpg', y='price', alpha=0.5)
    plt.title("Price vs MPG", fontsize=16, fontweight='bold')
    plt.xlabel("MPG", fontsize=12)
    plt.ylabel("Price (£)", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(actuals, predictions, model_name, R_SQUARE):
    """Scatter plot of actual vs predicted values for regression"""
    plt.figure(figsize=(10, 8))
    plt.scatter(actuals, predictions, alpha=0.4, c="teal", label="Predicted vs Actual", s=50)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)],
             "r--", label="Perfect Prediction (Actual = Predicted)", linewidth=2)
    plt.xlabel("Actual Price (£)", fontsize=12)
    plt.ylabel("Predicted Price (£)", fontsize=12)
    plt.title(f"Actual vs Predicted Price\nR² = {R_SQUARE:.3f} - {model_name}", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_residuals(actuals, predictions, model_name):
    """Plot residuals for regression analysis"""
    residuals = actuals - predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual plot
    axes[0].scatter(predictions, residuals, alpha=0.4, c='steelblue')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel("Predicted Price (£)", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title(f"Residual Plot - {model_name}", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_xlabel("Residuals", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Classification

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_classification_metrics(metrics_dict, title="Classification Metrics"):
    """Bar plot of classification metrics"""
    # Extract metrics (excluding confusion matrix)
    plot_metrics = {k: v for k, v in metrics_dict.items() if k != 'confusion_matrix'}
    
    plt.figure(figsize=(10, 6))
    metrics_names = list(plot_metrics.keys())
    metrics_values = list(plot_metrics.values())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(metrics_names, metrics_values, color=colors[:len(metrics_names)])
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(fontsize=11)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve"):
    """Plot ROC curve for binary classification"""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Clustering

def plot_clusters(df: pd.DataFrame, cluster_col: str = 'cluster_label',  x_col: str = 'mpg', y_col: str = 'price',  title: str = "Car Clusters"):
    """Scatter plot of clusters with labels"""
    plt.figure(figsize=(12, 8))
    
    # Use cluster_label if available, otherwise use cluster
    label_col = cluster_col if cluster_col in df.columns else 'cluster'
    
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=label_col,
        palette='Set2',
        s=100,
        alpha=0.6,
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_col.upper() if len(x_col) <= 3 else x_col.title(), fontsize=12)
    plt.ylabel(y_col.title(), fontsize=12)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cluster_distribution(df: pd.DataFrame, cluster_col: str = 'cluster_label'):
    """Bar plot showing distribution of samples across clusters"""
    plt.figure(figsize=(10, 6))
    
    label_col = cluster_col if cluster_col in df.columns else 'cluster'
    cluster_counts = df[label_col].value_counts().sort_index()
    
    colors = sns.color_palette('Set2', n_colors=len(cluster_counts))
    bars = plt.bar(range(len(cluster_counts)), cluster_counts.values, 
                   color=colors, edgecolor='black', linewidth=1.2)
    
    plt.xticks(range(len(cluster_counts)), cluster_counts.index, rotation=45, ha='right')
    plt.ylabel('Number of Cars', fontsize=12)
    plt.xlabel('Cluster', fontsize=12)
    plt.title('Distribution of Cars Across Clusters', fontsize=16, fontweight='bold')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_cluster_characteristics(df: pd.DataFrame, cluster_col: str = 'cluster_label', features: list = None):
    """Box plots showing characteristics of each cluster"""
    if features is None:
        features = ['price', 'mileage', 'mpg', 'engineSize']
    
    label_col = cluster_col if cluster_col in df.columns else 'cluster'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features[:4]):
        if feature in df.columns:
            sns.boxplot(data=df, x=label_col, y=feature, 
                       palette='Set2', ax=axes[idx])
            axes[idx].set_title(f'{feature.title()} by Cluster', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Cluster', fontsize=11)
            axes[idx].set_ylabel(feature.title(), fontsize=11)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_cluster_pairplot(df: pd.DataFrame, cluster_col: str = 'cluster_label',
                          features: list = None):
    """Pair plot showing relationships between features colored by cluster"""
    if features is None:
        features = ['price', 'mileage', 'mpg', 'engineSize']
    
    label_col = cluster_col if cluster_col in df.columns else 'cluster'
    
    # Select relevant columns
    plot_df = df[[label_col] + [f for f in features if f in df.columns]].copy()
    
    # Create pairplot
    g = sns.pairplot(plot_df, hue=label_col, palette='Set2', 
                     diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
    g.fig.suptitle('Feature Relationships by Cluster', 
                   fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

def plot_silhouette_comparison(clustering_results: dict):
    """Bar plot comparing silhouette scores across clustering algorithms"""
    algorithms = []
    scores = []
    
    for name, results in clustering_results.items():
        if results['silhouette_score'] is not None:
            algorithms.append(name.upper())
            scores.append(results['silhouette_score'])
    
    if not algorithms:
        print("No valid silhouette scores to plot")
        return
    
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(algorithms, scores, color=colors[:len(algorithms)], 
                   edgecolor='black', linewidth=1.2)
    
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.xlabel('Clustering Algorithm', fontsize=12)
    plt.title('Clustering Algorithm Comparison', fontsize=16, fontweight='bold')
    plt.ylim(-0.2, 1.0)
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# Feature Importance

def plot_feature_importances(values, feature_names, title):
    """Horizontal bar plot of feature importances/coefficients"""
    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": values
        })
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    plt.barh(importance_df["feature"], importance_df["importance"], 
             color=colors, edgecolor='black', linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# Association rules graphs

def plot_association_rules_scatter(rules_df, metric='lift', title='Association Rules'):
    """Scatter plot of support vs confidence colored by lift"""
    if rules_df.empty:
        print("No rules to plot")
        return
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(rules_df['support'], rules_df['confidence'], 
                         c=rules_df[metric], s=rules_df[metric]*20,
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, label=metric.title())
    plt.xlabel('Support', fontsize=12)
    plt.ylabel('Confidence', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_top_rules(rules_df, metric='lift', top_n=10, title='Top Association Rules'):
    """Bar plot of top association rules by specified metric"""
    if rules_df.empty:
        print("No rules to plot")
        return
    
    top_rules = rules_df.nlargest(top_n, metric).copy()
    top_rules['rule'] = top_rules.apply(
        lambda x: f"{list(x['antecedents'])[0]} → {list(x['consequents'])[0]}", 
        axis=1
    )
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_rules)))
    bars = plt.barh(range(len(top_rules)), top_rules[metric], 
                    color=colors, edgecolor='black', linewidth=1)
    
    plt.yticks(range(len(top_rules)), top_rules['rule'])
    plt.xlabel(metric.title(), fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()