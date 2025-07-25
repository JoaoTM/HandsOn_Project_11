import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def json_to_dataframe(json_path):
    """
    Convert mesh_distance_results.json to a pandas DataFrame with proper parsing of filenames
    """
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    records = []
    
    for original_key, original_data in results.items():
        original_file = os.path.basename(original_data.get('original', original_key))
        
        for method_name, method_data in original_data.items():
            if method_name in ['original', 'relative_path']:
                continue
            
            if isinstance(method_data, str) and method_data == 'error':
                records.append({
                    'original_file': original_file,
                    'method': method_name,
                    'points': None,
                    'sampling_method': None,
                    'point_noise_std': None,
                    'normal_noise_std': None,
                    'distance_type': 'error',
                    'distance_value': 'No matching PLY file found'
                })
                continue
            
            if not isinstance(method_data, dict):
                continue
                
            for recon_path, metrics in method_data.items():
                if not isinstance(metrics, dict):
                    continue
                
                # Initialize default values
                points = None
                sampling_method = None
                point_noise_std = None
                normal_noise_std = None
                
                # Parse filename if available
                if recon_path and isinstance(recon_path, str):
                    try:
                        filename = os.path.basename(recon_path)
                        parts = filename.split('_')
                        
                        # Expected format: {id}_{hash}_trimesh_{num}_p{points}_{sampling}_ps{point_noise}_ns{normal_noise}.ply
                        if len(parts) >= 8:
                            # The points count is always the 5th part (index 4)
                            points_part = parts[4]
                            if points_part.startswith('p'):
                                points = int(points_part[1:])
                            
                            # Sampling method is the 6th part (index 5)
                            sampling_part = parts[5]
                            sampling_method = 'poisson_disk' if sampling_part == 'pois' else 'uniform'
                            
                            # Point noise is the 7th part (index 6)
                            point_noise_part = parts[6]
                            if point_noise_part.startswith('ps'):
                                point_noise_std = float(point_noise_part[2:]) / 100
                            
                            # Normal noise is the 8th part (index 7, remove .ply)
                            normal_noise_part = parts[7].replace('.ply', '')
                            if normal_noise_part.startswith('ns'):
                                normal_noise_std = float(normal_noise_part[2:]) / 100
                            
                            print(f"Parsed: points={points}, sampling={sampling_method}, p_noise={point_noise_std}, n_noise={normal_noise_std}")
                    except Exception as e:
                        print(f"Error parsing {filename}: {str(e)}")
                        continue
                
                # Add metrics to records
                if 'chamfer_distance' in metrics:
                    records.append({
                        'original_file': original_file,
                        'method': method_name,
                        'points': points,
                        'sampling_method': sampling_method,
                        'point_noise_std': point_noise_std,
                        'normal_noise_std': normal_noise_std,
                        'distance_type': 'chamfer_distance',
                        'distance_value': metrics['chamfer_distance']
                    })
                
                if 'hausdorff_distance' in metrics:
                    records.append({
                        'original_file': original_file,
                        'method': method_name,
                        'points': points,
                        'sampling_method': sampling_method,
                        'point_noise_std': point_noise_std,
                        'normal_noise_std': normal_noise_std,
                        'distance_type': 'hausdorff_distance',
                        'distance_value': metrics['hausdorff_distance']
                    })
    
    # Create DataFrame
    column_order = [
        'original_file', 'method', 'points', 'sampling_method',
        'point_noise_std', 'normal_noise_std', 'distance_type', 'distance_value'
    ]
    
    df = pd.DataFrame(records)
    
    # Ensure all columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    
    return df[column_order]





def plot_distance_metrics_grid(df, title, point_noise_std=0.0,save_path = "",**kwargs):
    """
    Visualize distance metrics in 2x2 grid with PSR/SPSR on top and stochastic methods below.
    
    Args:
        df: DataFrame containing the metrics data
        sampling_method: Either 'poisson_disk' or 'uniform' to filter by
        point_noise_std: Point noise standard deviation to filter by
    """
    filtered = df

    
    """ &  ((df['original_file'] == "00010429_fc56088abf10474bba06f659_trimesh_004.ply")|
         (df['original_file'] == "00011000_8a21002f126e4425a811e70a_trimesh_004.ply")|
         (df['original_file'] == "00011171_db6e2de6f4ae4ec493ebe2aa_trimesh_047.ply")|
         (df['original_file'] == "00011563_26a622427a024bf3af381ee6_trimesh_014.ply")|
         (df['original_file'] == "00011602_c087f04c99464bf7ab2380c4_trimesh_000.ply"))"""
    if len(filtered) == 0:
        print(f"No data found for sampling_method={sampling_method}, point_noise_std={point_noise_std}")
        return
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Reconstruction Quality Metrics\n {title})', y=1.02)
    
    # Method groupings
    top_methods = ['PSR', 'SPSR']
    bottom_methods = ['StochasticPSR', 'StochasticPSR_wOneSolve']
    
    # Plot chamfer distance - top methods
    chamfer_top = filtered[
        (filtered['distance_type'] == 'chamfer_distance') &
        (filtered['method'].isin(top_methods))
    ]
    sns.lineplot(data=chamfer_top, hue="method", **kwargs,
                 marker='o', errorbar="sd", ax=axs[0, 0])
    axs[0, 0].set_title('Chamfer Distance (PSR/SPSR)')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel(kwargs['x'])
    axs[0, 0].set_ylabel('Log Chamfer Distance')
    axs[0, 0].grid(True)
    
    # Plot hausdorff distance - top methods
    hausdorff_top = filtered[
        (filtered['distance_type'] == 'hausdorff_distance') &
        (filtered['method'].isin(top_methods))
    ]
    sns.lineplot(data=hausdorff_top, hue="method", **kwargs,
                 marker='o', ax=axs[0, 1])
    axs[0, 1].set_title('Hausdorff Distance (PSR/SPSR)')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xlabel(kwargs['x'])
    axs[0, 1].set_ylabel('Log Hausdorff Distance')
    axs[0, 1].grid(True)
    
    # Plot chamfer distance - bottom methods
    chamfer_bottom = filtered[
        (filtered['distance_type'] == 'chamfer_distance') &
        (filtered['method'].isin(bottom_methods))
    ]
    sns.lineplot(data=chamfer_bottom, hue="method", **kwargs,
                 marker='o', errorbar="sd", ax=axs[1, 0])
    axs[1, 0].set_title('Chamfer Distance (Stochastic Variants)')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlabel(kwargs['x'])
    axs[1, 0].set_ylabel('Log Chamfer Distance')
    axs[1, 0].grid(True)
    
    # Plot hausdorff distance - bottom methods
    hausdorff_bottom = filtered[
        (filtered['distance_type'] == 'hausdorff_distance') &
        (filtered['method'].isin(bottom_methods))
    ]
    sns.lineplot(data=hausdorff_bottom, hue="method", **kwargs,
                 marker='o', errorbar="sd", ax=axs[1, 1])
    axs[1, 1].set_title('Hausdorff Distance (Stochastic Variants)')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel(kwargs['x'])
    axs[1, 1].set_ylabel('Log Hausdorff Distance')
    axs[1, 1].grid(True)
    


    if save_path != "":
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")  # YYYY.MM.DD_HH:MM format
        save_path = os.path.join(save_path, f"interpret_distances_{timestamp}.png")
        plt.savefig(save_path, 
               dpi=300,           # Higher resolution
               bbox_inches='tight', # Prevent cropping
               transparent=False) 
    plt.tight_layout(pad=2.0)  # Extra padding
    plt.show()


#df = json_to_dataframe(r"H:\Hands-on_AI_based_3D_Vision\Project_11\Sampled_Comparisons_Reconstructions\mesh_distance_results.json")


results_filepath=r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_100_Reconstruction\mesh_distance_results_.json"

df = json_to_dataframe(results_filepath)

print(df.head())
print("df.shape",df.shape)

save_path = os.path.dirname(results_filepath)
filtered = df[
        #(df['sampling_method'] == sampling_method) &
        (df['point_noise_std'] == 0.0) 
    
    ]

"""plot_distance_metrics_grid(filtered,title="point_noise_std= 0.0" ,save_path = save_path,
                           x='points', y='distance_value',style="sampling_method" ,units="original_file",estimator=None,)
"""
# point_noise_std points distance_value sampling_method original_file

filtered = df[ (df['sampling_method'] == "uniform")  ]

plot_distance_metrics_grid(filtered,title="sampling_method = uniform" ,save_path = save_path,
                           x='point_noise_std', y='distance_value',style="points" ,units="original_file",estimator=None,)

plot_distance_metrics_grid(filtered,title="sampling_method = uniform" ,save_path = save_path,
                           x='point_noise_std', y='distance_value',style="points")



"""# Filter the DataFrame
filtered_df = df[
    (df['points'] == 100) &
    (df['method'] == 'StochasticPSR_wOneSolve')
]

# Print the filtered results
print("Filtered Results:")
print(filtered_df[['original_file', 'method', 'points', 'sampling_method', 
                  'point_noise_std', 'normal_noise_std', 'distance_type', 
                  'distance_value']].to_string(index=False))

Attempting StochasticPSR_wOneSolve reconstruction...
Input points shape: (100, 3), normals shape: (100, 3)
Running StochasticPSR_wOneSolve with grid_res=80
Creating interpolator: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 977/977 [00:19<00:00, 49.22it/s]
Grid prepared with shape: (512000, 3)
Computing implicit function...
SDD:   0%|                                                                                                                                         | 0/1000 [00:00<?, ?it/s]
Failed to compute implicit function: Cannot take a larger sample (size 128) than population (size 100) when 'replace=False'
FAILED StochasticPSR_wOneSolve: Cannot take a larger sample (size 128) than population (size 100) when 'replace=False'"""

#_1000_unif_ps0001_ns0050