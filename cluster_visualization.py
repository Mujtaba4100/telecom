"""
Cluster Visualization with PCA
==============================
KMeans (k=6) and DBSCAN clustering with 2D/3D visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================
# LOAD AND PREPARE DATA
# ============================================

print("=" * 60)
print("CLUSTER VISUALIZATION WITH PCA")
print("=" * 60)

# Load golden table
df = pd.read_csv('golden_table.csv')
print(f"\n✓ Loaded golden_table.csv: {df.shape[0]:,} rows")

# Select features
features = [
    'voice_total_duration_mins',
    'voice_total_calls',
    'sms_total_messages',
    'data_total_mb',
    'intl_total_calls',
    'intl_total_duration_mins',
    'is_international_user'
]

X = df[features].fillna(0)
print(f"✓ Selected {len(features)} features")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Standardized features")

# ============================================
# KMEANS CLUSTERING (k=6)
# ============================================

print("\n" + "=" * 60)
print("KMEANS CLUSTERING (k=6)")
print("=" * 60)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['kmeans_cluster'] = kmeans_labels

print("\n✓ KMeans cluster distribution:")
for cluster in range(6):
    count = (kmeans_labels == cluster).sum()
    print(f"  Cluster {cluster}: {count:,} ({count/len(kmeans_labels)*100:.2f}%)")

# ============================================
# DBSCAN - DETERMINE EPS FROM K-DISTANCE
# ============================================

print("\n" + "=" * 60)
print("DBSCAN CLUSTERING (auto eps from K-Distance)")
print("=" * 60)

# Sample for faster K-distance calculation
sample_size = min(10000, len(X_scaled))
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_idx]

# K-distance plot
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_sample)
distances, _ = nn.kneighbors(X_sample)
distances = np.sort(distances[:, k-1])

# Find elbow point (where gradient changes significantly)
gradient = np.gradient(distances)
elbow_idx = np.argmax(gradient > np.percentile(gradient, 95))
eps_value = distances[elbow_idx]

# Ensure reasonable eps (between 0.3 and 2.0)
eps_value = max(0.3, min(2.0, eps_value))
print(f"\n✓ Determined eps from K-Distance plot: {eps_value:.3f}")

# Fit DBSCAN on sample first, then use KNN to assign full dataset
dbscan = DBSCAN(eps=eps_value, min_samples=5)
sample_labels = dbscan.fit_predict(X_sample)

# Use KNN to assign labels to full dataset
from sklearn.neighbors import KNeighborsClassifier
mask = sample_labels != -1
if mask.sum() > 0:
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_sample[mask], sample_labels[mask])
    dbscan_labels = knn_clf.predict(X_scaled)
else:
    dbscan_labels = np.full(len(X_scaled), -1)

df['dbscan_cluster'] = dbscan_labels

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = (dbscan_labels == -1).sum()

print(f"✓ DBSCAN found {n_clusters} clusters")
print(f"✓ Noise points: {n_noise:,} ({n_noise/len(dbscan_labels)*100:.2f}%)")

print("\n✓ DBSCAN cluster distribution:")
for cluster in sorted(set(dbscan_labels)):
    count = (dbscan_labels == cluster).sum()
    label = "Noise" if cluster == -1 else f"Cluster {cluster}"
    print(f"  {label}: {count:,} ({count/len(dbscan_labels)*100:.2f}%)")

# ============================================
# PCA - REDUCE TO 3 COMPONENTS
# ============================================

print("\n" + "=" * 60)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 60)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"\n✓ Reduced to 3 principal components")
print(f"✓ Explained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.2f}%")
print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Add PCA components to dataframe
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]
df['PC3'] = X_pca[:, 2]

# ============================================
# 2D SCATTER PLOTS
# ============================================

print("\n" + "=" * 60)
print("GENERATING 2D PLOTS")
print("=" * 60)

# Sample for plotting (full data is too large)
plot_sample_size = min(20000, len(df))
np.random.seed(42)
plot_idx = np.random.choice(len(df), plot_sample_size, replace=False)
df_plot = df.iloc[plot_idx].copy()

# KMeans 2D Plot
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    df_plot['PC1'], 
    df_plot['PC2'], 
    c=df_plot['kmeans_cluster'], 
    cmap='tab10', 
    alpha=0.6, 
    s=10
)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title('KMeans Clustering (k=6) - PCA 2D Visualization', fontsize=14)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster')
plt.tight_layout()
plt.savefig('kmeans_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: kmeans_2d.png")

# DBSCAN 2D Plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot noise points first (in gray)
noise_mask = df_plot['dbscan_cluster'] == -1
if noise_mask.any():
    ax.scatter(
        df_plot.loc[noise_mask, 'PC1'],
        df_plot.loc[noise_mask, 'PC2'],
        c='lightgray',
        alpha=0.3,
        s=5,
        label='Noise'
    )

# Plot clustered points
cluster_mask = df_plot['dbscan_cluster'] != -1
if cluster_mask.any():
    scatter = ax.scatter(
        df_plot.loc[cluster_mask, 'PC1'],
        df_plot.loc[cluster_mask, 'PC2'],
        c=df_plot.loc[cluster_mask, 'dbscan_cluster'],
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')

ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title(f'DBSCAN Clustering (eps={eps_value:.2f}) - PCA 2D Visualization', fontsize=14)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('dbscan_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: dbscan_2d.png")

# ============================================
# INTERACTIVE 3D PLOTS (PLOTLY)
# ============================================

print("\n" + "=" * 60)
print("GENERATING 3D INTERACTIVE PLOTS")
print("=" * 60)

# Calculate cluster characteristics for tooltips
def get_cluster_info(cluster_id, cluster_type='kmeans'):
    """Get human-readable cluster information"""
    col = f'{cluster_type}_cluster'
    cluster_data = df[df[col] == cluster_id]
    
    avg_voice = cluster_data['voice_total_duration_mins'].mean()
    avg_data = cluster_data['data_total_mb'].mean()
    avg_intl = cluster_data['intl_total_calls'].mean()
    count = len(cluster_data)
    pct = (count / len(df)) * 100
    
    # Determine user type
    if avg_intl > 0.5:
        user_type = "International Users"
    elif avg_data > 500:
        user_type = "Heavy Data Users"
    elif avg_voice > 20:
        user_type = "Heavy Voice Users"
    elif avg_data > 100:
        user_type = "Moderate Data Users"
    elif avg_voice > 5:
        user_type = "Moderate Voice Users"
    else:
        user_type = "Light Users"
    
    return {
        'type': user_type,
        'size': count,
        'pct': pct,
        'voice_avg': avg_voice,
        'data_avg': avg_data / 1024,  # Convert to GB
        'intl_avg': avg_intl
    }

# Prepare enhanced dataframe for plotting
df_plot['Cluster_Type'] = df_plot['kmeans_cluster'].apply(
    lambda x: get_cluster_info(x, 'kmeans')['type']
)
df_plot['Size'] = df_plot['kmeans_cluster'].apply(
    lambda x: f"{get_cluster_info(x, 'kmeans')['size']:,} users ({get_cluster_info(x, 'kmeans')['pct']:.1f}%)"
)
df_plot['Avg_Voice'] = df_plot['kmeans_cluster'].apply(
    lambda x: f"{get_cluster_info(x, 'kmeans')['voice_avg']:.1f} mins"
)
df_plot['Avg_Data'] = df_plot['kmeans_cluster'].apply(
    lambda x: f"{get_cluster_info(x, 'kmeans')['data_avg']:.2f} GB"
)
df_plot['Avg_Intl'] = df_plot['kmeans_cluster'].apply(
    lambda x: f"{get_cluster_info(x, 'kmeans')['intl_avg']:.2f} calls"
)

# KMeans 3D Plot with rich hover information
fig_kmeans = px.scatter_3d(
    df_plot,
    x='PC1',
    y='PC2',
    z='PC3',
    color='kmeans_cluster',
    color_continuous_scale='Viridis',
    title='<b>KMeans Customer Segmentation (6 Clusters)</b><br>' +
          '<sub>Interactive 3D View - Each point is a customer, colored by segment</sub>',
    labels={'kmeans_cluster': 'Cluster ID'},
    opacity=0.7,
    hover_data={
        'PC1': False,
        'PC2': False,
        'PC3': False,
        'kmeans_cluster': True,
        'Cluster_Type': True,
        'Size': True,
        'Avg_Voice': True,
        'Avg_Data': True,
        'Avg_Intl': True
    }
)
fig_kmeans.update_traces(marker=dict(size=3))
fig_kmeans.update_layout(
    scene=dict(
        xaxis_title='Principal Component 1 (32.9% variance)',
        yaxis_title='Principal Component 2 (21.9% variance)',
        zaxis_title='Principal Component 3 (14.3% variance)'
    ),
    width=1200,
    height=850,
    font=dict(size=11),
    title_font_size=16,
    annotations=[
        dict(
            text="<b>How to use:</b><br>" +
                 "• Rotate: Click and drag<br>" +
                 "• Zoom: Scroll<br>" +
                 "• Hover: See customer segment details<br>" +
                 "• Legend: Click to show/hide clusters",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10)
        )
    ]
)
fig_kmeans.write_html('kmeans_3d.html')
print("✓ Saved: kmeans_3d.html")

# DBSCAN 3D Plot - with noise highlighted and cluster info
df_plot['dbscan_cluster_str'] = df_plot['dbscan_cluster'].apply(
    lambda x: 'Noise' if x == -1 else f'Cluster {x}'
)
df_plot['DBSCAN_Type'] = df_plot['dbscan_cluster'].apply(
    lambda x: 'Outlier/Noise' if x == -1 else get_cluster_info(x, 'dbscan')['type']
)
df_plot['DBSCAN_Size'] = df_plot['dbscan_cluster'].apply(
    lambda x: 'N/A' if x == -1 else f"{get_cluster_info(x, 'dbscan')['size']:,} users ({get_cluster_info(x, 'dbscan')['pct']:.1f}%)"
)

# Create color mapping - noise in gray
unique_clusters = sorted(df_plot['dbscan_cluster'].unique())
colors = {}
color_scale = px.colors.qualitative.Set1
for i, c in enumerate(unique_clusters):
    if c == -1:
        colors[f'Noise'] = 'lightgray'
    else:
        colors[f'Cluster {c}'] = color_scale[i % len(color_scale)]

fig_dbscan = px.scatter_3d(
    df_plot,
    x='PC1',
    y='PC2',
    z='PC3',
    color='dbscan_cluster_str',
    color_discrete_map=colors,
    title=f'<b>DBSCAN Density-Based Clustering</b><br>' +
          f'<sub>Found {n_clusters} natural clusters (eps={eps_value:.2f}) - Gray points are noise</sub>',
    labels={'dbscan_cluster_str': 'Cluster'},
    opacity=0.7,
    hover_data={
        'PC1': False,
        'PC2': False,
        'PC3': False,
        'dbscan_cluster_str': True,
        'DBSCAN_Type': True,
        'DBSCAN_Size': True
    }
)

# Make noise points smaller
fig_dbscan.for_each_trace(
    lambda t: t.update(marker=dict(size=2)) if t.name == 'Noise' else t.update(marker=dict(size=4))
)

fig_dbscan.update_layout(
    scene=dict(
        xaxis_title='Principal Component 1 (32.9% variance)',
        yaxis_title='Principal Component 2 (21.9% variance)',
        zaxis_title='Principal Component 3 (14.3% variance)'
    ),
    width=1200,
    height=850,
    font=dict(size=11),
    title_font_size=16,
    annotations=[
        dict(
            text="<b>How to use:</b><br>" +
                 "• Rotate: Click and drag<br>" +
                 "• Zoom: Scroll<br>" +
                 "• Hover: See cluster characteristics<br>" +
                 "• Legend: Click to show/hide clusters<br>" +
                 "<br><b>About DBSCAN:</b><br>" +
                 "Finds naturally dense groups<br>" +
                 "Gray = outliers/unusual patterns",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10)
        )
    ]
)
fig_dbscan.write_html('dbscan_3d.html')
print("✓ Saved: dbscan_3d.html")

# ============================================
# SAVE K-DISTANCE PLOT
# ============================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(distances, linewidth=2)
ax.axhline(y=eps_value, color='r', linestyle='--', label=f'Selected eps={eps_value:.3f}')
ax.axvline(x=elbow_idx, color='g', linestyle='--', alpha=0.5, label=f'Elbow point')
ax.set_xlabel('Points (sorted by distance)', fontsize=12)
ax.set_ylabel('5-th Nearest Neighbor Distance', fontsize=12)
ax.set_title('K-Distance Graph for DBSCAN eps Selection', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('k_distance_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: k_distance_plot.png")

# ============================================
# SAVE UPDATED DATAFRAME
# ============================================

df.to_csv('golden_table_clustered.csv', index=False)
print(f"\n✓ Saved updated dataframe: golden_table_clustered.csv")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Files Generated:
  - kmeans_2d.png       : 2D scatter plot (KMeans)
  - dbscan_2d.png       : 2D scatter plot (DBSCAN)
  - kmeans_3d.html      : Interactive 3D plot (KMeans)
  - dbscan_3d.html      : Interactive 3D plot (DBSCAN)
  - k_distance_plot.png : K-Distance plot for eps selection
  - golden_table_clustered.csv : Data with cluster labels

Clustering Results:
  - KMeans: 6 clusters
  - DBSCAN: {n_clusters} clusters + {n_noise:,} noise points
  
PCA Variance Explained:
  - PC1: {pca.explained_variance_ratio_[0]*100:.1f}%
  - PC2: {pca.explained_variance_ratio_[1]*100:.1f}%
  - PC3: {pca.explained_variance_ratio_[2]*100:.1f}%
  - Total: {sum(pca.explained_variance_ratio_)*100:.1f}%
""")

print("=" * 60)
print("Done! Open the .html files in a browser for interactive 3D plots.")
print("=" * 60)
