"""
Customer Segmentation Analysis
==============================
This script builds a Golden Table and implements clustering-based customer 
segmentation with proper validation and business-level insights.

Author: Data Analytics Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Performance settings
SAMPLE_SIZE = 10000  # Sample size for parameter tuning (faster)
DBSCAN_SAMPLE_SIZE = 5000  # Smaller sample for DBSCAN (very slow algorithm)
RANDOM_STATE = 42

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)


# ============================================
# PHASE 1 — DATA INTEGRATION
# ============================================

def load_datasets(subscriber_path, international_path):
    """
    Load both datasets from CSV files.
    
    Parameters:
    -----------
    subscriber_path : str
        Path to merged_subscriber_data.csv
    international_path : str
        Path to international_calls.csv
    
    Returns:
    --------
    tuple : (subscriber_df, international_df)
    """
    print("=" * 60)
    print("PHASE 1: DATA INTEGRATION")
    print("=" * 60)
    
    subscriber_df = pd.read_csv(subscriber_path)
    international_df = pd.read_csv(international_path)
    
    print(f"\n✓ Loaded subscriber data: {subscriber_df.shape[0]:,} rows, {subscriber_df.shape[1]} columns")
    print(f"✓ Loaded international data: {international_df.shape[0]:,} rows, {international_df.shape[1]} columns")
    
    return subscriber_df, international_df


def create_golden_table(subscriber_df, international_df, output_path):
    """
    Merge datasets and create the golden table with is_international_user flag.
    
    Parameters:
    -----------
    subscriber_df : DataFrame
        Main subscriber data
    international_df : DataFrame
        International calls data
    output_path : str
        Path to save golden_table.csv
    
    Returns:
    --------
    DataFrame : Merged golden table
    """
    # Get list of subscribers who made international calls
    international_subscribers = set(international_df['subscriberid'].unique())
    
    # Perform LEFT JOIN on subscriberid
    golden_table = subscriber_df.merge(
        international_df, 
        on='subscriberid', 
        how='left'
    )
    
    # Create is_international_user column
    golden_table['is_international_user'] = golden_table['subscriberid'].apply(
        lambda x: 1 if x in international_subscribers else 0
    )
    
    # Identify international-related columns (from international_df)
    intl_columns = [col for col in international_df.columns if col != 'subscriberid']
    
    # Fill missing international-related columns with 0
    for col in intl_columns:
        if golden_table[col].dtype in ['float64', 'int64']:
            golden_table[col] = golden_table[col].fillna(0)
        else:
            golden_table[col] = golden_table[col].fillna('')
    
    # Save the merged dataset
    golden_table.to_csv(output_path, index=False)
    
    print(f"\n✓ Created golden table: {golden_table.shape[0]:,} rows, {golden_table.shape[1]} columns")
    print(f"✓ International users: {golden_table['is_international_user'].sum():,} ({golden_table['is_international_user'].mean()*100:.2f}%)")
    print(f"✓ Saved to: {output_path}")
    
    return golden_table


# ============================================
# PHASE 2 — FEATURE SELECTION
# ============================================

def select_features(golden_table):
    """
    Select only behavioral numeric features for clustering.
    
    Parameters:
    -----------
    golden_table : DataFrame
        The merged golden table
    
    Returns:
    --------
    tuple : (X DataFrame, feature_names list)
    """
    print("\n" + "=" * 60)
    print("PHASE 2: FEATURE SELECTION")
    print("=" * 60)
    
    # Define the behavioral numeric features
    selected_features = [
        'voice_total_duration_mins',
        'voice_total_calls',
        'sms_total_messages',
        'data_total_mb',
        'intl_total_calls',
        'intl_total_duration_mins',
        'is_international_user'
    ]
    
    # Check which features exist in the dataframe
    available_features = [f for f in selected_features if f in golden_table.columns]
    missing_features = [f for f in selected_features if f not in golden_table.columns]
    
    if missing_features:
        print(f"\n⚠ Warning: Missing features: {missing_features}")
    
    X = golden_table[available_features].copy()
    
    print(f"\n✓ Selected {len(available_features)} behavioral features:")
    for f in available_features:
        print(f"  - {f}")
    
    return X, available_features


# ============================================
# PHASE 3 — DATA CLEANING
# ============================================

def clean_data(X, cap_outliers=True, percentile=99):
    """
    Clean the feature data: handle nulls and optionally cap outliers.
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    cap_outliers : bool
        Whether to cap extreme outliers
    percentile : int
        Percentile value for outlier capping
    
    Returns:
    --------
    DataFrame : Cleaned feature data
    """
    print("\n" + "=" * 60)
    print("PHASE 3: DATA CLEANING")
    print("=" * 60)
    
    # Check for null values
    null_counts = X.isnull().sum()
    total_nulls = null_counts.sum()
    
    print(f"\n✓ Null values check:")
    if total_nulls > 0:
        print(f"  Found {total_nulls} null values")
        for col in null_counts[null_counts > 0].index:
            print(f"  - {col}: {null_counts[col]} nulls")
    else:
        print("  No null values found")
    
    # Replace nulls with 0
    X = X.fillna(0)
    print(f"✓ Replaced all null values with 0")
    
    # Cap extreme outliers using percentile
    if cap_outliers:
        print(f"\n✓ Capping outliers at {percentile}th percentile:")
        for col in X.columns:
            if col != 'is_international_user':  # Don't cap binary column
                upper_limit = X[col].quantile(percentile / 100)
                outliers_count = (X[col] > upper_limit).sum()
                if outliers_count > 0:
                    X[col] = X[col].clip(upper=upper_limit)
                    print(f"  - {col}: capped {outliers_count} values at {upper_limit:.2f}")
    
    # Print descriptive statistics
    print("\n✓ Descriptive Statistics after cleaning:")
    print("-" * 40)
    print(X.describe().T.to_string())
    
    return X


# ============================================
# PHASE 4 — SCALING
# ============================================

def scale_features(X):
    """
    Standardize features using StandardScaler.
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    
    Returns:
    --------
    tuple : (X_scaled array, scaler object)
    """
    print("\n" + "=" * 60)
    print("PHASE 4: SCALING")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n✓ Applied StandardScaler to {X.shape[1]} features")
    print(f"✓ Scaled data shape: {X_scaled.shape}")
    print(f"✓ Mean after scaling: ~{X_scaled.mean():.6f} (should be ~0)")
    print(f"✓ Std after scaling: ~{X_scaled.std():.6f} (should be ~1)")
    
    return X_scaled, scaler


# ============================================
# PHASE 5 — KMEANS CLUSTERING
# ============================================

def find_optimal_k(X_scaled, k_range=(2, 11)):
    """
    Find optimal number of clusters using elbow method and silhouette score.
    Uses MiniBatchKMeans and sampling for fast execution on large datasets.
    
    Parameters:
    -----------
    X_scaled : array
        Scaled feature data
    k_range : tuple
        Range of k values to test (start, end)
    
    Returns:
    --------
    tuple : (optimal_k, inertias, silhouette_scores)
    """
    print("\n" + "=" * 60)
    print("PHASE 5: KMEANS CLUSTERING")
    print("=" * 60)
    
    # Use sampling for silhouette score calculation (much faster)
    n_samples = min(SAMPLE_SIZE, len(X_scaled))
    if len(X_scaled) > SAMPLE_SIZE:
        print(f"\n✓ Using {n_samples:,} sample for parameter tuning (full data: {len(X_scaled):,})")
        np.random.seed(RANDOM_STATE)
        sample_idx = np.random.choice(len(X_scaled), n_samples, replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled
    
    k_values = range(k_range[0], k_range[1])
    inertias = []
    silhouette_scores_list = []
    
    print("\n✓ Running MiniBatchKMeans for k = 2 to 10:")
    print("-" * 50)
    print(f"{'k':<5} {'Inertia':<20} {'Silhouette Score':<15}")
    print("-" * 50)
    
    for k in k_values:
        # MiniBatchKMeans is much faster than regular KMeans
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, 
                                  batch_size=1024, n_init=3, max_iter=100)
        kmeans.fit(X_scaled)
        
        inertia = kmeans.inertia_
        # Calculate silhouette on sample for speed
        sample_labels = kmeans.predict(X_sample)
        silhouette = silhouette_score(X_sample, sample_labels)
        
        inertias.append(inertia)
        silhouette_scores_list.append(silhouette)
        
        print(f"{k:<5} {inertia:<20.2f} {silhouette:<15.4f}")
    
    # Find optimal k based on silhouette score
    optimal_k = k_values[np.argmax(silhouette_scores_list)]
    
    print("-" * 50)
    print(f"\n✓ Optimal k based on silhouette score: {optimal_k}")
    print(f"  Best silhouette score: {max(silhouette_scores_list):.4f}")
    
    return optimal_k, list(k_values), inertias, silhouette_scores_list


def plot_elbow_curve(k_values, inertias, silhouette_scores, optimal_k):
    """
    Plot elbow curve and silhouette scores.
    
    Parameters:
    -----------
    k_values : list
        List of k values tested
    inertias : list
        Inertia values for each k
    silhouette_scores : list
        Silhouette scores for each k
    optimal_k : int
        The chosen optimal k
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method for Optimal k', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score for Each k', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved elbow curve plot: kmeans_optimization.png")


def fit_kmeans(X_scaled, optimal_k):
    """
    Fit final MiniBatchKMeans model with optimal k.
    
    Parameters:
    -----------
    X_scaled : array
        Scaled feature data
    optimal_k : int
        Optimal number of clusters
    
    Returns:
    --------
    array : Cluster labels
    """
    kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, 
                              batch_size=1024, n_init=3, max_iter=100)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    print(f"\n✓ Fitted final MiniBatchKMeans model with k={optimal_k}")
    print(f"✓ Cluster distribution:")
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  - Cluster {cluster}: {count:,} ({count/len(kmeans_labels)*100:.2f}%)")
    
    return kmeans_labels


# ============================================
# PHASE 6 — DBSCAN CLUSTERING
# ============================================

def estimate_eps(X_scaled, k=5):
    """
    Estimate eps parameter using k-nearest neighbors distance plot.
    Uses sampling for fast estimation on large datasets.
    
    Parameters:
    -----------
    X_scaled : array
        Scaled feature data
    k : int
        Number of neighbors to consider
    
    Returns:
    --------
    float : Suggested eps value
    """
    print("\n" + "=" * 60)
    print("PHASE 6: DBSCAN CLUSTERING")
    print("=" * 60)
    
    # Use smaller sampling for faster eps estimation
    n_samples = min(DBSCAN_SAMPLE_SIZE, len(X_scaled))
    if len(X_scaled) > DBSCAN_SAMPLE_SIZE:
        print(f"\n✓ Using {n_samples:,} sample for eps estimation")
        np.random.seed(RANDOM_STATE)
        sample_idx = np.random.choice(len(X_scaled), n_samples, replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled
    
    # Fit nearest neighbors on sample
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_sample)
    distances, _ = nn.kneighbors(X_sample)
    
    # Sort distances to the k-th nearest neighbor
    distances = np.sort(distances[:, k-1])
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances, linewidth=2)
    plt.xlabel('Points (sorted by distance)', fontsize=12)
    plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
    plt.title('K-Distance Graph for DBSCAN eps Estimation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('dbscan_eps_estimation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved k-distance plot: dbscan_eps_estimation.png")
    
    # Estimate eps using elbow detection (simple heuristic)
    gradient = np.gradient(distances)
    elbow_idx = np.argmax(gradient > np.percentile(gradient, 90))
    suggested_eps = distances[elbow_idx]
    
    return suggested_eps


def tune_dbscan(X_scaled, eps_range, min_samples_range):
    """
    Tune DBSCAN parameters and find best configuration.
    Uses sampling for fast parameter tuning on large datasets.
    
    Parameters:
    -----------
    X_scaled : array
        Scaled feature data
    eps_range : list
        List of eps values to try
    min_samples_range : list
        List of min_samples values to try
    
    Returns:
    --------
    tuple : (best_eps, best_min_samples, best_score)
    """
    # Use smaller sampling for faster DBSCAN tuning
    n_samples = min(DBSCAN_SAMPLE_SIZE, len(X_scaled))
    if len(X_scaled) > DBSCAN_SAMPLE_SIZE:
        print(f"\n✓ Using {n_samples:,} sample for DBSCAN tuning")
        np.random.seed(RANDOM_STATE)
        sample_idx = np.random.choice(len(X_scaled), n_samples, replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled
    
    print(f"\n✓ Tuning DBSCAN parameters:")
    print(f"  - eps range: {eps_range}")
    print(f"  - min_samples range: {min_samples_range}")
    
    best_score = -1
    best_eps = eps_range[0]
    best_min_samples = min_samples_range[0]
    best_n_clusters = 0
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_sample)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)
            
            # Only compute silhouette if we have valid clusters
            if n_clusters >= 2 and noise_ratio < 0.5:
                # Exclude noise points for silhouette calculation
                mask = labels != -1
                if mask.sum() > n_clusters:
                    # Use subset of non-noise points for silhouette
                    score = silhouette_score(X_sample[mask], labels[mask])
                else:
                    score = -1
            else:
                score = -1
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'silhouette': score
            })
            
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
                best_n_clusters = n_clusters
    
    # Print results table
    print("\n" + "-" * 70)
    print(f"{'eps':<8} {'min_samp':<10} {'clusters':<10} {'noise':<10} {'noise%':<10} {'silhouette':<12}")
    print("-" * 70)
    for r in results:
        sil_str = f"{r['silhouette']:.4f}" if r['silhouette'] > 0 else "N/A"
        print(f"{r['eps']:<8.2f} {r['min_samples']:<10} {r['n_clusters']:<10} {r['n_noise']:<10} {r['noise_ratio']*100:<10.1f} {sil_str:<12}")
    print("-" * 70)
    
    print(f"\n✓ Best DBSCAN parameters:")
    print(f"  - eps: {best_eps}")
    print(f"  - min_samples: {best_min_samples}")
    print(f"  - Silhouette score: {best_score:.4f}")
    print(f"  - Number of clusters: {best_n_clusters}")
    
    return best_eps, best_min_samples


def fit_dbscan(X_scaled, eps, min_samples):
    """
    Fit DBSCAN with chosen parameters.
    
    Parameters:
    -----------
    X_scaled : array
        Scaled feature data
    eps : float
        DBSCAN eps parameter
    min_samples : int
        DBSCAN min_samples parameter
    
    Returns:
    --------
    array : Cluster labels
    """
    # For large datasets, use approximate DBSCAN with sampling + nearest neighbor assignment
    n_full = len(X_scaled)
    if n_full > 50000:
        # Train DBSCAN on a sample
        sample_size = min(20000, n_full)
        print(f"\n✓ Training DBSCAN on {sample_size:,} sample, then assigning {n_full:,} points...")
        np.random.seed(RANDOM_STATE)
        sample_idx = np.random.choice(n_full, sample_size, replace=False)
        X_sample = X_scaled[sample_idx]
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        sample_labels = dbscan.fit_predict(X_sample)
        
        # Use nearest neighbors to assign remaining points
        from sklearn.neighbors import KNeighborsClassifier
        # Train classifier only on clustered (non-noise) points
        mask = sample_labels != -1
        if mask.sum() > 0:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_sample[mask], sample_labels[mask])
            
            # Predict for all points
            dbscan_labels = knn.predict(X_scaled)
        else:
            # All noise - assign -1 to everyone
            dbscan_labels = np.full(n_full, -1)
    else:
        print(f"\n✓ Fitting DBSCAN on full dataset ({n_full:,} samples)...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"\n✓ Fitted DBSCAN model:")
    print(f"  - Number of clusters (excluding noise): {n_clusters}")
    print(f"  - Number of noise points (-1 label): {n_noise:,} ({n_noise/len(dbscan_labels)*100:.2f}%)")
    
    print(f"\n✓ DBSCAN Cluster distribution:")
    unique, counts = np.unique(dbscan_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        label = f"Noise" if cluster == -1 else f"Cluster {cluster}"
        print(f"  - {label}: {count:,} ({count/len(dbscan_labels)*100:.2f}%)")
    
    return dbscan_labels, n_clusters, n_noise


# ============================================
# PHASE 7 — CLUSTER ANALYSIS
# ============================================

def analyze_clusters(golden_table, feature_names):
    """
    Analyze clusters and compare with manual *_lover columns.
    
    Parameters:
    -----------
    golden_table : DataFrame
        Golden table with cluster labels
    feature_names : list
        List of feature names used for clustering
    
    Returns:
    --------
    dict : Analysis results
    """
    print("\n" + "=" * 60)
    print("PHASE 7: CLUSTER ANALYSIS")
    print("=" * 60)
    
    lover_columns = ['call_lover', 'data_lover', 'upload_lover', 'download_lover']
    
    results = {}
    
    # KMeans Cluster Analysis
    print("\n" + "=" * 50)
    print("KMEANS CLUSTER ANALYSIS")
    print("=" * 50)
    
    kmeans_summary = golden_table.groupby('kmeans_cluster')[feature_names].mean()
    print("\n✓ Mean feature values by KMeans cluster:")
    print(kmeans_summary.T.to_string())
    
    results['kmeans_summary'] = kmeans_summary
    
    # DBSCAN Cluster Analysis
    print("\n" + "=" * 50)
    print("DBSCAN CLUSTER ANALYSIS")
    print("=" * 50)
    
    dbscan_summary = golden_table.groupby('dbscan_cluster')[feature_names].mean()
    print("\n✓ Mean feature values by DBSCAN cluster:")
    print(dbscan_summary.T.to_string())
    
    results['dbscan_summary'] = dbscan_summary
    
    # Compare with manual *_lover columns
    print("\n" + "=" * 50)
    print("COMPARISON WITH MANUAL SEGMENTATION")
    print("=" * 50)
    
    # KMeans matching percentages
    print("\n✓ KMeans clusters vs *_lover flags:")
    print("-" * 60)
    
    kmeans_matching = {}
    for cluster in sorted(golden_table['kmeans_cluster'].unique()):
        cluster_data = golden_table[golden_table['kmeans_cluster'] == cluster]
        cluster_size = len(cluster_data)
        
        print(f"\nCluster {cluster} (n={cluster_size:,}):")
        matching = {}
        for lover in lover_columns:
            if lover in golden_table.columns:
                pct = cluster_data[lover].mean() * 100
                matching[lover] = pct
                print(f"  - {lover}: {pct:.1f}% are {lover}")
        kmeans_matching[cluster] = matching
    
    results['kmeans_matching'] = kmeans_matching
    
    # DBSCAN matching percentages (excluding noise)
    print("\n✓ DBSCAN clusters vs *_lover flags:")
    print("-" * 60)
    
    dbscan_matching = {}
    for cluster in sorted(golden_table['dbscan_cluster'].unique()):
        cluster_data = golden_table[golden_table['dbscan_cluster'] == cluster]
        cluster_size = len(cluster_data)
        
        label = "Noise" if cluster == -1 else f"Cluster {cluster}"
        print(f"\n{label} (n={cluster_size:,}):")
        matching = {}
        for lover in lover_columns:
            if lover in golden_table.columns:
                pct = cluster_data[lover].mean() * 100
                matching[lover] = pct
                print(f"  - {lover}: {pct:.1f}% are {lover}")
        dbscan_matching[cluster] = matching
    
    results['dbscan_matching'] = dbscan_matching
    
    return results


def identify_cluster_types(kmeans_summary, kmeans_matching):
    """
    Identify which cluster corresponds to which user type.
    
    Parameters:
    -----------
    kmeans_summary : DataFrame
        Mean features by cluster
    kmeans_matching : dict
        Matching percentages with lover columns
    
    Returns:
    --------
    dict : Cluster type assignments
    """
    print("\n" + "=" * 50)
    print("CLUSTER TYPE IDENTIFICATION")
    print("=" * 50)
    
    cluster_types = {}
    
    for cluster in kmeans_summary.index:
        row = kmeans_summary.loc[cluster]
        matches = kmeans_matching.get(cluster, {})
        
        # Determine cluster type based on characteristics
        characteristics = []
        
        # Check for high voice usage
        if row.get('voice_total_calls', 0) > kmeans_summary['voice_total_calls'].mean():
            characteristics.append("High Voice")
        
        # Check for high data usage
        if row.get('data_total_mb', 0) > kmeans_summary['data_total_mb'].mean():
            characteristics.append("High Data")
        
        # Check for international users
        if row.get('is_international_user', 0) > 0.1:
            characteristics.append("International")
        
        # Check for high SMS usage
        if row.get('sms_total_messages', 0) > kmeans_summary['sms_total_messages'].mean():
            characteristics.append("High SMS")
        
        # Determine primary type based on lover matching
        primary_type = "General User"
        max_match = 0
        for lover, pct in matches.items():
            if pct > max_match and pct > 20:  # Threshold of 20%
                max_match = pct
                primary_type = lover.replace('_lover', '').replace('_', ' ').title() + " User"
        
        if not characteristics:
            characteristics.append("Low Usage")
        
        cluster_types[cluster] = {
            'primary_type': primary_type,
            'characteristics': characteristics,
            'matching': matches
        }
        
        print(f"\nCluster {cluster}:")
        print(f"  Primary Type: {primary_type}")
        print(f"  Characteristics: {', '.join(characteristics)}")
    
    return cluster_types


# ============================================
# PHASE 8 — PACKAGE GENERATION
# ============================================

def round_to_business_value(value, options):
    """
    Round a value to the nearest business-friendly option.
    
    Parameters:
    -----------
    value : float
        Value to round
    options : list
        List of business-friendly values
    
    Returns:
    --------
    numeric : Nearest business-friendly value
    """
    if value <= 0:
        return options[0]
    return min(options, key=lambda x: abs(x - value))


def generate_personalized_packages(cluster_id, cluster_summary, cluster_types):
    """
    Generate personalized telecom packages based on cluster characteristics.
    
    Parameters:
    -----------
    cluster_id : int
        Cluster ID to generate package for
    cluster_summary : DataFrame
        Mean feature values by cluster
    cluster_types : dict
        Cluster type information
    
    Returns:
    --------
    dict : Personalized package details
    """
    # Get cluster stats
    stats = cluster_summary.loc[cluster_id]
    cluster_info = cluster_types.get(cluster_id, {})
    
    # Apply +20% to average usage
    multiplier = 1.20
    
    # Business-friendly value options
    data_options = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]  # GB
    call_options = [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]  # minutes
    sms_options = [0, 50, 100, 200, 500, 1000, 2000]  # messages
    intl_options = [0, 10, 20, 30, 50, 100, 200]  # minutes
    
    # Calculate package values based on average usage + 20%
    data_target = stats.get('data_total_mb', 0) * multiplier / 1024  # Convert MB to GB
    call_target = stats.get('voice_total_duration_mins', 0) * multiplier
    sms_target = stats.get('sms_total_messages', 0) * multiplier
    intl_target = stats.get('intl_total_duration_mins', 0) * multiplier
    is_intl = stats.get('is_international_user', 0) > 0.05
    
    # Round to business-friendly values
    data_package = round_to_business_value(data_target, data_options)
    call_package = round_to_business_value(call_target, call_options)
    sms_package = round_to_business_value(sms_target, sms_options)
    intl_package = round_to_business_value(intl_target, intl_options) if is_intl else 0
    
    package = {
        'cluster_id': cluster_id,
        'cluster_type': cluster_info.get('primary_type', 'General User'),
        'characteristics': cluster_info.get('characteristics', []),
        'data_package_gb': data_package,
        'call_minutes_package': call_package,
        'sms_package': sms_package,
        'international_addon_mins': intl_package,
        'includes_international': is_intl,
        'avg_data_mb': stats.get('data_total_mb', 0),
        'avg_voice_mins': stats.get('voice_total_duration_mins', 0),
        'avg_sms': stats.get('sms_total_messages', 0),
        'avg_intl_mins': stats.get('intl_total_duration_mins', 0)
    }
    
    return package


def generate_all_packages(cluster_summary, cluster_types):
    """
    Generate personalized packages for all clusters.
    
    Parameters:
    -----------
    cluster_summary : DataFrame
        Mean feature values by cluster
    cluster_types : dict
        Cluster type information
    
    Returns:
    --------
    list : List of package dictionaries
    """
    print("\n" + "=" * 60)
    print("PHASE 8: PACKAGE GENERATION")
    print("=" * 60)
    
    packages = []
    
    for cluster_id in cluster_summary.index:
        package = generate_personalized_packages(cluster_id, cluster_summary, cluster_types)
        packages.append(package)
    
    print("\n✓ Generated personalized packages for all clusters:\n")
    print("=" * 80)
    
    for pkg in packages:
        print(f"\n{'='*60}")
        print(f"CLUSTER {pkg['cluster_id']}: {pkg['cluster_type']}")
        print(f"{'='*60}")
        print(f"Characteristics: {', '.join(pkg['characteristics'])}")
        print(f"\n📦 RECOMMENDED PACKAGE:")
        print(f"  ├─ Data Package: {pkg['data_package_gb']} GB")
        print(f"  ├─ Call Minutes: {pkg['call_minutes_package']} mins")
        print(f"  ├─ SMS Package: {pkg['sms_package']} messages")
        if pkg['includes_international']:
            print(f"  └─ International Add-on: {pkg['international_addon_mins']} mins ✈️")
        else:
            print(f"  └─ International Add-on: Not included")
        print(f"\n📊 Based on cluster averages:")
        print(f"  ├─ Avg Data: {pkg['avg_data_mb']:.2f} MB")
        print(f"  ├─ Avg Voice: {pkg['avg_voice_mins']:.2f} mins")
        print(f"  ├─ Avg SMS: {pkg['avg_sms']:.2f}")
        print(f"  └─ Avg Int'l: {pkg['avg_intl_mins']:.2f} mins")
    
    return packages


# ============================================
# FINAL OUTPUT SUMMARY
# ============================================

def print_final_summary(optimal_k, n_dbscan_clusters, n_noise, cluster_types, 
                       kmeans_matching, packages, kmeans_summary):
    """
    Print comprehensive final summary.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "FINAL ANALYSIS SUMMARY" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # 1. Optimal K
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ 1. OPTIMAL K CHOSEN                                         │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   KMeans optimal clusters: {optimal_k}")
    
    # 2. DBSCAN clusters
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ 2. DBSCAN CLUSTERING RESULTS                                │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   Number of clusters (excluding noise): {n_dbscan_clusters}")
    print(f"   Number of noise points: {n_noise:,}")
    
    # 3. Cluster behavioral summary
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ 3. CLUSTER BEHAVIORAL SUMMARY                               │")
    print("└─────────────────────────────────────────────────────────────┘")
    for cluster, info in cluster_types.items():
        print(f"\n   Cluster {cluster}: {info['primary_type']}")
        print(f"   Characteristics: {', '.join(info['characteristics'])}")
    
    # 4. Matching percentage with manual segmentation
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ 4. MATCHING % WITH MANUAL SEGMENTATION                      │")
    print("└─────────────────────────────────────────────────────────────┘")
    for cluster, matches in kmeans_matching.items():
        print(f"\n   Cluster {cluster}:")
        for lover, pct in matches.items():
            marker = "★" if pct > 30 else " "
            print(f"   {marker} {lover}: {pct:.1f}%")
    
    # 5. Personalized packages
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│ 5. SUGGESTED PERSONALIZED TELECOM PACKAGES                  │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    for pkg in packages:
        print(f"\n   ┌─ Cluster {pkg['cluster_id']}: {pkg['cluster_type']} ─────────────")
        print(f"   │  📱 Data: {pkg['data_package_gb']} GB")
        print(f"   │  📞 Calls: {pkg['call_minutes_package']} mins")
        print(f"   │  💬 SMS: {pkg['sms_package']} msgs")
        if pkg['includes_international']:
            print(f"   │  ✈️  Int'l: {pkg['international_addon_mins']} mins")
        print(f"   └─────────────────────────────────────────────")
    
    print("\n" + "═" * 80)
    print("Analysis complete!")
    print("═" * 80)


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Main function to orchestrate the entire analysis pipeline.
    """
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "CUSTOMER SEGMENTATION ANALYSIS PIPELINE" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # File paths
    subscriber_path = 'merged_subscriber_data.csv'
    international_path = 'international_calls.csv'
    golden_table_path = 'golden_table.csv'
    
    # ========================================
    # PHASE 1: Data Integration
    # ========================================
    subscriber_df, international_df = load_datasets(subscriber_path, international_path)
    golden_table = create_golden_table(subscriber_df, international_df, golden_table_path)
    
    # ========================================
    # PHASE 2: Feature Selection
    # ========================================
    X, feature_names = select_features(golden_table)
    
    # ========================================
    # PHASE 3: Data Cleaning
    # ========================================
    X = clean_data(X, cap_outliers=True, percentile=99)
    
    # ========================================
    # PHASE 4: Scaling
    # ========================================
    X_scaled, scaler = scale_features(X)
    
    # ========================================
    # PHASE 5: KMeans Clustering
    # ========================================
    optimal_k, k_values, inertias, silhouette_scores = find_optimal_k(X_scaled)
    plot_elbow_curve(k_values, inertias, silhouette_scores, optimal_k)
    kmeans_labels = fit_kmeans(X_scaled, optimal_k)
    golden_table['kmeans_cluster'] = kmeans_labels
    
    # ========================================
    # PHASE 6: DBSCAN Clustering
    # ========================================
    suggested_eps = estimate_eps(X_scaled)
    
    # Define eps and min_samples ranges based on data characteristics
    # Simplified eps and min_samples ranges for faster tuning
    eps_range = [0.5, 1.0, 1.5]
    min_samples_range = [5, 10]
    
    best_eps, best_min_samples = tune_dbscan(X_scaled, eps_range, min_samples_range)
    dbscan_labels, n_dbscan_clusters, n_noise = fit_dbscan(X_scaled, best_eps, best_min_samples)
    golden_table['dbscan_cluster'] = dbscan_labels
    
    # ========================================
    # PHASE 7: Cluster Analysis
    # ========================================
    analysis_results = analyze_clusters(golden_table, feature_names)
    cluster_types = identify_cluster_types(
        analysis_results['kmeans_summary'], 
        analysis_results['kmeans_matching']
    )
    
    # ========================================
    # PHASE 8: Package Generation
    # ========================================
    packages = generate_all_packages(analysis_results['kmeans_summary'], cluster_types)
    
    # ========================================
    # Save final golden table with cluster labels
    # ========================================
    golden_table.to_csv(golden_table_path, index=False)
    print(f"\n✓ Updated golden_table.csv with cluster labels")
    
    # ========================================
    # FINAL OUTPUT SUMMARY
    # ========================================
    print_final_summary(
        optimal_k=optimal_k,
        n_dbscan_clusters=n_dbscan_clusters,
        n_noise=n_noise,
        cluster_types=cluster_types,
        kmeans_matching=analysis_results['kmeans_matching'],
        packages=packages,
        kmeans_summary=analysis_results['kmeans_summary']
    )
    
    # Return results dictionary
    return {
        'optimal_k': optimal_k,
        'n_dbscan_clusters': n_dbscan_clusters,
        'n_noise': n_noise,
        'cluster_types': cluster_types,
        'kmeans_matching': analysis_results['kmeans_matching'],
        'dbscan_matching': analysis_results['dbscan_matching'],
        'packages': packages,
        'golden_table': golden_table
    }


if __name__ == "__main__":
    results = main()
