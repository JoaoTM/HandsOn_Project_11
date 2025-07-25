import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def json_to_dataframe(json_path):
    """
    Convert timing results JSON to a pandas DataFrame with columns:
    original_file, method, points, sampling_method, point_noise_std, normal_noise_std, time, status
    """
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    records = []
    
    for entry in results:
        # Initialize default values
        record = {
            'original_file': None,
            'method': entry.get('method'),
            'points': None,
            'sampling_method': None,
            'point_noise_std': None,
            'normal_noise_std': None,
            'time': entry.get('time'),
            'status': entry.get('status')
        }
        
        # Parse filename if available
        input_path = entry.get('input_path')
        if input_path:
            try:
                filename = os.path.basename(input_path)
                parts = filename.split('_')
                
                # Expected format: {id}_{hash}_trimesh_{num}_p{points}_{sampling}_ps{point_noise}_ns{normal_noise}.ply
                if len(parts) >= 8:
                    # Extract base original filename (first 4 parts joined with _)
                    record['original_file'] = '_'.join(parts[:4]) + '.ply'
                    
                    # Extract points (e.g., from 'p128')
                    points_part = parts[4]
                    if points_part.startswith('p'):
                        record['points'] = int(points_part[1:])
                    
                    # Extract sampling method
                    sampling_part = parts[5]
                    record['sampling_method'] = 'poisson_disk' if sampling_part == 'pois' else 'uniform'
                    
                    # Extract point noise (e.g., 'ps0000' -> 0.00)
                    point_noise_part = parts[6]
                    if point_noise_part.startswith('ps'):
                        record['point_noise_std'] = float(point_noise_part[2:]) / 100
                    
                    # Extract normal noise (e.g., 'ns0000' -> 0.00)
                    normal_noise_part = parts[7].replace('.ply', '')
                    if normal_noise_part.startswith('ns'):
                        record['normal_noise_std'] = float(normal_noise_part[2:]) / 100
                    
            except Exception as e:
                print(f"Error parsing {filename}: {str(e)}")
        
        records.append(record)
    
    # Create DataFrame with consistent column order
    column_order = [
        'original_file', 'method', 'points', 'sampling_method',
        'point_noise_std', 'normal_noise_std', 'time', 'status'
    ]
    
    df = pd.DataFrame(records)
    
    # Ensure all columns exist (fill missing with None)
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    
    return df[column_order]

# Example usage
times_filepath = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_100_Reconstruction\times_.json"
df_times = json_to_dataframe(times_filepath)


print(df_times.shape)
# Display the first few rows
print(df_times.head())

# Group by method and calculate mean time
time_means = df_times.groupby('method')['time'].mean().reset_index()
time_means = time_means.sort_values('time')
print(time_means)

time_means = df_times.groupby('points')['time'].mean().reset_index()
time_means = time_means.sort_values('time')
print(time_means)

time_means = df_times.groupby('normal_noise_std')['time'].mean().reset_index()
time_means = time_means.sort_values('time')
print(time_means)




time_pivot = df_times.pivot_table(
    values='time',
    index='method',
    columns='points',
    aggfunc='mean'
)
print(time_pivot)

time_pivot = df_times.pivot_table(
    values='time',
    index='method',
    columns='normal_noise_std',
    aggfunc='mean'
)
print("\n\n",time_pivot)

time_pivot = df_times.pivot_table(
    values='time',
    index='points',
    columns='normal_noise_std',
    aggfunc='mean'
)
print("\n\n",time_pivot)


time_pivot = df_times.pivot_table(
    values='time',
    index='method',
    columns='sampling_method',
    aggfunc='mean'
)
print("\n\n",time_pivot)


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 6))

# Create lineplot with error bars
sns.lineplot(
    data=df_times,
    x='points',
    y='time',
    hue='method',
    style='method',
    markers=True,
    markersize=8,
    dashes=False,
    palette='viridis',
    errorbar='sd',  # Show standard deviation
    err_style='band',  # Error as shaded bands
    ax=ax
)

# Customize the plot
ax.set_title('Reconstruction Time by Point Count and Method', fontsize=14)
ax.set_xlabel('Number of Points', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_yscale('log')
ax.set_xticks(df_times['points'].unique())  # Ensure all point counts are shown
ax.grid(True, linestyle='--', alpha=0.6)

# Adjust legend
ax.legend(
    title='Method',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0
)

plt.tight_layout()
plt.show()

def doPCA(df_times):
    # Prepare numeric data for PCA
    numeric_cols = ['points', 'point_noise_std', 'time']
    df_numeric = df_times[numeric_cols].dropna()

    # One-hot encode categorical variables
    df_categorical = pd.get_dummies(df_times[['method', 'sampling_method']])
    df_combined = pd.concat([df_numeric, df_categorical], axis=1).dropna()

    # Separate features and target
    X = df_combined.drop('time', axis=1)
    y = df_combined['time']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]))
    principal_components = pca.fit_transform(X_scaled)

    # Create results DataFrame
    pca_results = pd.DataFrame({
        'Feature': X.columns,
        'PC1_Loading': pca.components_[0],
        'PC2_Loading': pca.components_[1],
        'Variance_Explained': pca.explained_variance_ratio_
    })

    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), 
            pca.explained_variance_ratio_,
            alpha=0.5,
            align='center')
    plt.step(range(1, len(pca.explained_variance_ratio_)+1),
            np.cumsum(pca.explained_variance_ratio_),
            where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Variance Explained by Principal Components')
    plt.show()

    # Create a heatmap of component loadings - CORRECTED VERSION
    plt.figure(figsize=(12, 8))
    n_components_to_show = min(5, len(pca.components_))  # Show up to 5 components
    sns.heatmap(pca.components_[:n_components_to_show, :].T,  # Transpose for better visualization
                cmap='coolwarm',
                yticklabels=X.columns,
                xticklabels=['PC'+str(x) for x in range(1, n_components_to_show+1)],
                cbar_kws={'label': 'Component Loading'})
    plt.title('PCA Component Loadings')
    plt.tight_layout()
    plt.show()

    # Print most influential features
    print("\nMost influential features on time:")
    print(pca_results.sort_values('PC1_Loading', key=abs, ascending=False)[['Feature', 'PC1_Loading']].head(10))

# Run PCA on selected columns
doPCA(df_times[['original_file', 'method', 'points', 'sampling_method', 'point_noise_std', 'time']])