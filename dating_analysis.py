#!/usr/bin/env python3
# Dating History Analysis Script
# This script analyzes dating history data to determine which factors contribute most to relationship satisfaction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Function to load and preprocess data
def load_data(filepath='history.csv'):
    """
    Load the dating history dataset and prepare it for analysis
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with {df.shape[0]} relationships and {df.shape[1]} attributes")
    return df

# Function for exploratory data analysis
def exploratory_analysis(df):
    """
    Perform basic exploratory analysis on the dating history dataset
    """
    print("\n=== Exploratory Data Analysis ===")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe().round(2))
    
    # Correlation with relationship satisfaction
    target = 'Overall_Relationship_Satisfaction'
    correlations = df.corr()[target].sort_values(ascending=False)
    print("\nFactors most correlated with relationship satisfaction:")
    print(correlations.drop(target).round(2))
    
    # Plotting correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Dating Attributes')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("Correlation matrix saved as 'correlation_matrix.png'")
    
    # Top 5 factors correlation plot
    top_factors = correlations.drop(target).nlargest(5).index
    plt.figure(figsize=(12, 8))
    for factor in top_factors:
        plt.scatter(df[factor], df[target], alpha=0.7, label=factor)
    
    plt.xlabel('Factor Value')
    plt.ylabel('Relationship Satisfaction')
    plt.title('Top 5 Factors vs. Relationship Satisfaction')
    plt.legend()
    plt.grid(True)
    plt.savefig('top_factors_scatter.png')
    print("Top factors scatter plot saved as 'top_factors_scatter.png'")
    
    return top_factors

# Function to run multiple regression models
def regression_analysis(df):
    """
    Run various regression models to determine which factors best predict relationship satisfaction
    """
    print("\n=== Regression Analysis ===")
    
    # Prepare data
    target = 'Overall_Relationship_Satisfaction'
    features = df.drop(['Relationship_ID', target], axis=1)
    X = features
    y = df[target]
    
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features.columns)
    
    # Run OLS regression with statsmodels for detailed statistics
    X_with_const = sm.add_constant(X_scaled_df)
    model = sm.OLS(y, X_with_const).fit()
    print("\nOLS Regression Results Summary:")
    print(model.summary().tables[1])
    
    # Extract coefficients and create dataframe for visualization
    coefs = pd.DataFrame({
        'Feature': X_with_const.columns[1:],  # Skip the constant
        'Coefficient': model.params[1:],
        'P-value': model.pvalues[1:],
        'Significance': ['**' if p < 0.05 else '*' if p < 0.1 else '' for p in model.pvalues[1:]]
    })
    coefs['AbsCoef'] = abs(coefs['Coefficient'])
    coefs = coefs.sort_values('AbsCoef', ascending=False)
    
    print("\nFeature importance based on absolute coefficient size:")
    print(coefs[['Feature', 'Coefficient', 'P-value', 'Significance']].round(3))
    
    # Create coefficient plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Coefficient', y='Feature', data=coefs, 
                    palette=['darkred' if c < 0 else 'darkgreen' for c in coefs['Coefficient']])
    
    # Add significance markers
    for i, p in enumerate(coefs['P-value']):
        if p < 0.05:
            ax.text(0, i, "  **", ha='left', va='center', fontweight='bold')
        elif p < 0.1:
            ax.text(0, i, "  *", ha='left', va='center')
    
    plt.title('Feature Importance (Coefficient Size)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
    plt.xlabel('Standardized Coefficient')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")
    
    # Run different regression models and compare
    print("\nComparing different regression models:")
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1)
    }
    
    results = {}
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        mse = -cv_scores.mean()
        rmse = np.sqrt(mse)
        
        # Fit the model
        model.fit(X_scaled, y)
        
        # Store results
        if hasattr(model, 'coef_'):
            coefs_dict = dict(zip(features.columns, model.coef_))
            top_features = sorted(coefs_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        else:
            top_features = []
        
        results[name] = {
            'RMSE': rmse,
            'Top Features': top_features
        }
        
        print(f"\n{name}:")
        print(f"  RMSE: {rmse:.4f}")
        if top_features:
            print("  Top 5 features and coefficients:")
            for feature, coef in top_features:
                print(f"    {feature}: {coef:.4f}")
    
    return coefs, results

# Function for PCA analysis to identify groups of correlated features
def pca_analysis(df):
    """
    Perform Principal Component Analysis to identify groups of correlated dating attributes
    """
    print("\n=== Principal Component Analysis (PCA) ===")
    
    # Prepare data
    target = 'Overall_Relationship_Satisfaction'
    features = df.drop(['Relationship_ID', target], axis=1)
    X = features.values
    feature_names = features.columns
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    print("\nRunning PCA to identify underlying dating attribute dimensions...")
    pca = PCA()
    pca.fit(X_scaled)
    
    # Get the explained variance ratio
    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance)
    
    # Determine optimal number of components (explaining 80% variance)
    n_components = np.argmax(cumulative_variance >= 80) + 1
    
    print(f"\nExplained variance by component:")
    for i, var in enumerate(explained_variance[:5]):
        print(f"Component {i+1}: {var:.2f}%")
    
    print(f"\nOptimal number of components (explaining at least 80% variance): {n_components}")
    
    # Visualize explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='skyblue')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance')
    plt.axhline(y=80, color='r', linestyle='--', label='80% Explained Variance Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance by Principal Components')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pca_explained_variance.png')
    print("PCA explained variance plot saved as 'pca_explained_variance.png'")
    
    # Use the optimal number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Interpret the principal components
    components = pd.DataFrame(pca.components_, columns=feature_names)
    
    # Get the feature loadings for each component
    print("\nFeature loadings for each principal component:")
    for i in range(n_components):
        print(f"\nPrincipal Component {i+1} (explains {explained_variance[i]:.2f}% of variance):")
        # Get the absolute loadings and sort
        loadings = components.iloc[i].abs().sort_values(ascending=False)
        # Print top contributing features (loading > 0.3 is typically considered significant)
        significant_features = loadings[loadings > 0.3]
        if len(significant_features) > 0:
            print("  Significant features (|loading| > 0.3):")
            for feature, loading in significant_features.items():
                # Get the actual sign of the loading (not absolute value)
                actual_loading = components.iloc[i][feature]
                sign = "+" if actual_loading > 0 else "-"
                print(f"    {feature}: {sign}{loading:.3f}")
        else:
            print("  No features with loadings > 0.3")
    
    # Visualize the first two principal components
    plt.figure(figsize=(12, 10))
    plt.subplot(111)
    # Plot feature loadings
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, pca.components_[0, i]*8, pca.components_[1, i]*8, 
                 head_width=0.1, head_length=0.1, fc='lightblue', ec='blue')
        plt.text(pca.components_[0, i]*8.2, pca.components_[1, i]*8.2, feature, color='black')
    
    # Plot the transformed data points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df[target], cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='Relationship Satisfaction')
    
    # Add circle
    circle = plt.Circle((0, 0), 8, fill=False, linestyle='--', color='gray', alpha=0.3)
    plt.gca().add_patch(circle)
    
    # Add titles and labels
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
    plt.title('Feature Loadings and Data Points in the First Two Principal Components')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_biplot.png')
    print("PCA biplot saved as 'pca_biplot.png'")
    
    # Identify groups of correlated features using the loadings
    print("\nIdentifying groups of correlated features:")
    
    # Create a correlation matrix of the features
    feature_corr = features.corr().abs()
    
    # Find highly correlated feature pairs (correlation > 0.6)
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if feature_corr.iloc[i, j] > 0.6:
                high_corr_pairs.append((feature_names[i], feature_names[j], feature_corr.iloc[i, j]))
    
    # Sort by correlation strength
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (correlation > 0.6):")
        for f1, f2, corr in high_corr_pairs:
            print(f"  {f1} & {f2}: {corr:.3f}")
    else:
        print("\nNo feature pairs with correlation > 0.6 found.")
    
    # Attempt to identify feature groups using clustering on components
    # Use the loadings from all components for clustering
    loadings_for_clustering = components.T.values
    
    # Determine optimal number of clusters using silhouette method or elbow method
    # For simplicity, we'll use between 2-5 clusters
    wcss = []
    for i in range(2, 6):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(loadings_for_clustering)
        wcss.append(kmeans.inertia_)
    
    # Find the "elbow" point
    optimal_clusters = np.argmin(np.diff(np.diff(wcss))) + 2
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(loadings_for_clustering)
    
    # Group features by cluster
    feature_clusters = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in feature_clusters:
            feature_clusters[cluster_id] = []
        feature_clusters[cluster_id].append(feature_names[i])
    
    print(f"\nFeature clusters (using K-means with {optimal_clusters} clusters):")
    for cluster_id, features_in_cluster in feature_clusters.items():
        print(f"\nCluster {cluster_id+1}:")
        for feature in features_in_cluster:
            print(f"  - {feature}")
    
    return pca, components, feature_clusters

# Function for advanced clustering analysis
def advanced_clustering(df, feature_clusters):
    """
    Perform advanced clustering analyses on the dating dataset
    """
    print("\n=== Advanced Clustering Analysis ===")
    
    # Prepare data
    target = 'Overall_Relationship_Satisfaction'
    features = df.drop(['Relationship_ID', target], axis=1)
    feature_names = features.columns
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Part 1: Hierarchical Clustering of Features
    print("\n1. Hierarchical Clustering of Dating Attributes")
    
    # Calculate correlations and convert to a distance matrix
    corr = features.corr()
    # Convert correlation to distance (1 - |corr|)
    distance = 1 - np.abs(corr)
    
    # Perform hierarchical clustering
    Z = linkage(squareform(distance), 'ward')
    
    # Plot dendrogram
    plt.figure(figsize=(14, 10))
    plt.title('Hierarchical Clustering of Dating Attributes')
    plt.xlabel('Dating Attributes')
    plt.ylabel('Distance (1 - |Correlation|)')
    dendrogram(
        Z,
        labels=feature_names,
        orientation='top',
        leaf_rotation=90,
        color_threshold=0.3 * max(Z[:, 2])
    )
    plt.tight_layout()
    plt.savefig('hierarchical_clustering_features.png')
    print("Hierarchical clustering dendrogram saved as 'hierarchical_clustering_features.png'")
    
    # Cut the tree to get clusters
    n_clusters = 3  # Try with 3 clusters
    # Fix the clustering parameters - using ward linkage which works with euclidean distance
    cluster = AgglomerativeClustering(n_clusters=n_clusters)
    # Convert distance matrix back to a feature matrix for clustering
    # We'll use the original scaled features
    cluster_labels = cluster.fit_predict(X_scaled.T)  # Transpose to cluster features
    
    # Group features by cluster
    feature_groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in feature_groups:
            feature_groups[label] = []
        feature_groups[label].append(feature_names[i])
    
    print(f"\nFeature groups from hierarchical clustering (n_clusters={n_clusters}):")
    for group_id, features_in_group in feature_groups.items():
        print(f"\nGroup {group_id+1} - Features:")
        for feature in features_in_group:
            print(f"  - {feature}")
    
    # Part 2: Clustering Your Relationships (Dating Types)
    print("\n2. Clustering of Relationships (Dating Types)")
    
    # We're now clustering the relationships, not the features
    # Use PCA to reduce dimensionality first
    pca = PCA(n_components=3)  # Try with top 3 components
    X_pca = pca.fit_transform(X_scaled)
    
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    for n in range(2, min(6, df.shape[0])):  # Don't try more clusters than relationships
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append((n, score))
    
    # Choose cluster number with highest silhouette score
    optimal_n = max(silhouette_scores, key=lambda x: x[1])[0]
    print(f"\nOptimal number of relationship clusters (best silhouette score): {optimal_n}")
    
    # Perform K-means with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_n, random_state=42)
    relationship_clusters = kmeans.fit_predict(X_pca)
    
    # Add cluster labels to the original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['RelationshipCluster'] = relationship_clusters
    
    # Analyze the clusters
    print(f"\nCharacteristics of each relationship cluster:")
    for i in range(optimal_n):
        cluster_data = df_with_clusters[df_with_clusters['RelationshipCluster'] == i]
        print(f"\nCluster {i+1} ({len(cluster_data)} relationships):")
        print(f"  Average Satisfaction: {cluster_data[target].mean():.2f}")
        
        # Top 5 distinguishing features (highest values)
        top_features = cluster_data.drop(['Relationship_ID', target, 'RelationshipCluster'], axis=1).mean().sort_values(ascending=False).head(5)
        print("  Top 5 highest attributes:")
        for feat, val in top_features.items():
            print(f"    - {feat}: {val:.2f}")
        
        # Bottom 5 distinguishing features (lowest values)
        bottom_features = cluster_data.drop(['Relationship_ID', target, 'RelationshipCluster'], axis=1).mean().sort_values().head(5)
        print("  Top 5 lowest attributes:")
        for feat, val in bottom_features.items():
            print(f"    - {feat}: {val:.2f}")
    
    # Visualize relationship clusters in PCA space
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=relationship_clusters, cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(scatter, label='Relationship Cluster')
    
    # Add relationship IDs as labels
    for i, txt in enumerate(df['Relationship_ID']):
        plt.annotate(int(txt), (X_pca[i, 0], X_pca[i, 1]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Relationship Clusters in PCA Space')
    plt.grid(True, alpha=0.3)
    plt.savefig('relationship_clusters.png')
    print("Relationship clusters plot saved as 'relationship_clusters.png'")
    
    # Part 3: DBSCAN for Identifying Core Feature Groups (Robust to Outliers)
    print("\n3. DBSCAN Clustering for Robust Feature Grouping")
    
    # Transpose the data to cluster features instead of relationships
    X_features = X_scaled.T
    
    # Determine epsilon by using k-distance graph
    neighbors = 3  # for small feature set
    
    # Calculate distances
    distances = pdist(X_features)
    distances = squareform(distances)
    
    # Sort distances for each point
    k_dist = np.sort(distances, axis=1)[:, neighbors]
    
    # Find a suitable epsilon from the "elbow" in k-distance graph
    k_dist_sorted = np.sort(k_dist)
    
    # Simple heuristic to find the elbow point
    diffs = np.diff(k_dist_sorted)
    elbow_index = np.argmax(diffs) + 1
    epsilon = k_dist_sorted[elbow_index]
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=2)
    dbscan_labels = dbscan.fit_predict(X_features)
    
    # Group features by cluster
    dbscan_groups = {}
    for i, label in enumerate(dbscan_labels):
        if label == -1:  # Noise points
            if 'Outliers' not in dbscan_groups:
                dbscan_groups['Outliers'] = []
            dbscan_groups['Outliers'].append(feature_names[i])
        else:
            if label not in dbscan_groups:
                dbscan_groups[label] = []
            dbscan_groups[label].append(feature_names[i])
    
    print(f"\nFeature groups from DBSCAN clustering (eps={epsilon:.2f}, min_samples=2):")
    for group_name, features_in_group in dbscan_groups.items():
        if group_name == 'Outliers':
            if features_in_group:
                print(f"\nOutlier Features (unique characteristics):")
                for feature in features_in_group:
                    print(f"  - {feature}")
        else:
            print(f"\nCore Feature Group {group_name+1}:")
            for feature in features_in_group:
                print(f"  - {feature}")
    
    # Part 4: Combined visualization of all clustering results
    # Create a table showing feature assignments across different clustering methods
    clustering_results = pd.DataFrame(index=feature_names)
    
    # Add hierarchical clustering results
    hierarchical_labels = {}
    for group_id, features in feature_groups.items():
        for feature in features:
            hierarchical_labels[feature] = f"Group {group_id+1}"
    clustering_results['Hierarchical'] = pd.Series(hierarchical_labels)
    
    # Add DBSCAN results
    dbscan_result_labels = {}
    for group_name, features in dbscan_groups.items():
        label = "Outlier" if group_name == 'Outliers' else f"Group {group_name+1}"
        for feature in features:
            dbscan_result_labels[feature] = label
    clustering_results['DBSCAN'] = pd.Series(dbscan_result_labels)
    
    # Add Kmeans from PCA results
    if feature_clusters:
        feature_clusters_formatted = {}
        for cluster_id, features in feature_clusters.items():
            for feature in features:
                feature_clusters_formatted[feature] = f"Cluster {cluster_id+1}"
        clustering_results['K-Means'] = pd.Series(feature_clusters_formatted)
    
    print("\nComparison of feature clustering results across methods:")
    print(clustering_results)
    
    # Save the clustering comparison to CSV
    clustering_results.to_csv('feature_clustering_comparison.csv')
    print("Feature clustering comparison saved to 'feature_clustering_comparison.csv'")
    
    return feature_groups, dbscan_groups, df_with_clusters

# Function to generate dating insights and recommendations
def generate_insights(df, coefs, results):
    """
    Generate insights and recommendations based on the analysis
    """
    print("\n=== Dating Insights and Recommendations ===")
    
    # Get top positive and negative factors
    pos_factors = coefs[coefs['Coefficient'] > 0].head(3)['Feature'].tolist()
    neg_factors = coefs[coefs['Coefficient'] < 0].head(3)['Feature'].tolist()
    
    print("\nTop 3 factors positively associated with relationship satisfaction:")
    for factor in pos_factors:
        print(f"- {factor}")
    
    print("\nTop 3 factors negatively associated with relationship satisfaction:")
    for factor in neg_factors:
        print(f"- {factor}")
    
    # Calculate optimal weights for dating criteria
    total_pos_coef = coefs[coefs['Coefficient'] > 0]['AbsCoef'].sum()
    weights = {}
    
    for _, row in coefs[coefs['Coefficient'] > 0].iterrows():
        weights[row['Feature']] = round(row['AbsCoef'] / total_pos_coef * 100, 2)
    
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    print("\nRecommended weights for your dating criteria (based on positive factors):")
    for factor, weight in sorted_weights:
        print(f"- {factor}: {weight}%")
    
    # Check for relationship patterns
    high_sat = df[df['Overall_Relationship_Satisfaction'] > 0.7]
    low_sat = df[df['Overall_Relationship_Satisfaction'] < 0.4]
    
    print(f"\nCharacteristics of your {len(high_sat)} most satisfying relationships:")
    if len(high_sat) > 0:
        for factor in high_sat.drop(['Relationship_ID', 'Overall_Relationship_Satisfaction'], axis=1).mean().sort_values(ascending=False).head(5).index:
            print(f"- High {factor}: {high_sat[factor].mean():.2f}")
    
    print(f"\nCharacteristics of your {len(low_sat)} least satisfying relationships:")
    if len(low_sat) > 0:
        for factor in low_sat.drop(['Relationship_ID', 'Overall_Relationship_Satisfaction'], axis=1).mean().sort_values().head(5).index:
            print(f"- Low {factor}: {low_sat[factor].mean():.2f}")
    
    return weights

# Main function
def main():
    """
    Main function to run the analysis pipeline
    """
    print("=== Dating History Analysis ===")
    
    # Load data
    df = load_data()
    
    # Exploratory analysis
    top_factors = exploratory_analysis(df)
    
    # Regression analysis
    coefs, results = regression_analysis(df)
    
    # PCA analysis to identify correlated feature groups
    pca, components, feature_clusters = pca_analysis(df)
    
    # Advanced clustering analysis
    feature_groups, dbscan_groups, df_with_clusters = advanced_clustering(df, feature_clusters)
    
    # Generate insights
    weights = generate_insights(df, coefs, results)
    
    print("\nAnalysis complete! Check the generated plots for visualizations.")
    print("You can use the weights provided above to guide your future dating decisions.")
    print("The clustering analyses have identified different types of relationships and feature groupings.")

if __name__ == "__main__":
    main() 