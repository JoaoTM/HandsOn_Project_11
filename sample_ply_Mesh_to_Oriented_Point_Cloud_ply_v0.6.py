import numpy as np
import open3d as o3d
import os
from open3d.visualization import draw_geometries
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

import os
from pathlib import Path
import re



def sample_points_from_mesh(input_mesh_path, output_ply_path, num_points=500, 
                          sampling_method='poisson_disk', 
                          point_noise_mean=0.0, point_noise_std=0.0,
                          normal_noise_mean=0.0, normal_noise_std=0.0):
    """
    Samples points from a 3D mesh and saves them as a point cloud with normals.
    
    Args:
        input_mesh_path (str): Path to the input mesh file (.ply)
        output_ply_path (str): Path to save the output point cloud (.ply)
        num_points (int): Number of points to sample (default: 500)
        sampling_method (str): Sampling method ('poisson_disk', 'uniform', 'random') (default: 'poisson_disk')
        point_noise_mean (float): Mean of Gaussian noise added to point positions (default: 0.0)
        point_noise_std (float): Standard deviation of Gaussian noise added to point positions (default: 0.0)
        normal_noise_mean (float): Mean of Gaussian noise added to normals (angular noise in radians) (default: 0.0)
        normal_noise_std (float): Standard deviation of Gaussian noise added to normals (angular noise in radians) (default: 0.0)
        
    Returns:
        bool: True if operation succeeded, False otherwise
    """
    try:
        # Load ground-truth mesh
        print("Loading mesh...")
        mesh = o3d.io.read_triangle_mesh(input_mesh_path)
        
        # Verify mesh loaded correctly
        if not mesh.has_vertices():
            raise ValueError("Mesh has no vertices")
            
        print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")

        # Compute vertex normals (if not already in file)
        if not mesh.has_vertex_normals():
            print("Computing vertex normals...")
            mesh.compute_vertex_normals()

        # Sample points using selected method
        print(f"Sampling {num_points} points using {sampling_method} method...")
        if sampling_method == 'poisson_disk':
            pcd = mesh.sample_points_poisson_disk(
                number_of_points=num_points, 
                use_triangle_normal=True
            )
        elif sampling_method == 'uniform':
            pcd = mesh.sample_points_uniformly(
                number_of_points=num_points,
                use_triangle_normal=True
            )
        elif sampling_method == 'random':
            pcd = mesh.sample_points_random_uniform(
                number_of_points=num_points,
                use_triangle_normal=True
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Verify we got normals
        if not pcd.has_normals():
            raise ValueError("Sampled point cloud has no normals")

        # Add Gaussian noise to points if specified
        if point_noise_std > 0:
            print(f"Adding Gaussian noise to points (mean={point_noise_mean}, std={point_noise_std})")
            points = np.asarray(pcd.points)
            noise = np.random.normal(point_noise_mean, point_noise_std, points.shape)
            pcd.points = o3d.utility.Vector3dVector(points + noise)
        
        # Add Gaussian noise to normals if specified
        if normal_noise_std > 0:
            print(f"Adding Gaussian noise to normals (mean={normal_noise_mean}, std={normal_noise_std})")
            normals = np.asarray(pcd.normals)
            
            # Generate random angles (in radians) using normal distribution
            angles = np.random.normal(normal_noise_mean, normal_noise_std, len(normals))
            
            # Create random rotation vectors perpendicular to original normals
            perp_vectors = np.cross(normals, np.random.rand(*normals.shape))
            perp_vectors /= np.linalg.norm(perp_vectors, axis=1)[:, np.newaxis]
            
            # Rotate normals by the random angles
            noisy_normals = (normals * np.cos(angles)[:, np.newaxis] + 
                           perp_vectors * np.sin(angles)[:, np.newaxis])
            
            # Ensure they remain unit vectors
            noisy_normals /= np.linalg.norm(noisy_normals, axis=1)[:, np.newaxis]
            pcd.normals = o3d.utility.Vector3dVector(noisy_normals)

        # Save oriented point cloud
        print(f"Saving to {output_ply_path}...")
        success = o3d.io.write_point_cloud(
            filename=output_ply_path,
            pointcloud=pcd,
            write_ascii=True
        )
        
        if not success:
            raise RuntimeError("Failed to write point cloud")
            
        print(f"Success! Sampled {len(pcd.points)} points with normals to {output_ply_path}")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def visualize_with_normals(ply_path1, ply_path2, output_image_path=None, window_name="Comparison"):
    """
    Visualizes a mesh and point cloud with:
    - Mesh in gray
    - Points as red spheres
    - Normals as blue lines (length = 2*sphere_radius)
    """
    try:
        # Load geometries
        mesh = o3d.io.read_triangle_mesh(ply_path1)
        pcd = o3d.io.read_point_cloud(ply_path2)
        
        # Validate loaded data
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh has no vertices")
        if len(pcd.points) == 0:
            raise ValueError("Point cloud is empty")
        if not pcd.has_normals():
            raise ValueError("Point cloud has no normals")

        # Compute normals if needed
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Color the mesh
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray

        # Calculate optimal sphere radius
        sphere_radius = calculate_optimal_sphere_radius(mesh, pcd)
        
        # Create spheres and normal lines
        spheres, normal_lines = create_spheres_and_normals(pcd, sphere_radius)
        
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1200, height=1200)
        
        # Add geometries
        vis.add_geometry(mesh)
        for sphere in spheres:
            vis.add_geometry(sphere)
        vis.add_geometry(normal_lines)
        
        # Adjust view
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        combined_center = (mesh.get_center() + pcd.get_center()) / 2
        ctr.set_lookat(combined_center)
        ctr.set_zoom(0.8)

        print(f"Visualizing: Mesh with {len(mesh.vertices)} vertices and "
              f"Point Cloud with {len(pcd.points)} points")
        print(f"Sphere radius: {sphere_radius:.6f}")
        print(f"Normal length: {2*sphere_radius:.6f}")

        # Run visualization
        vis.run()
        
        # Save image if requested
        if output_image_path:
            vis.capture_screen_image(output_image_path)
            print(f"Comparison image saved to {output_image_path}")
        
        vis.destroy_window()
        return True
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return False

def create_spheres_and_normals(pcd, radius):
    """Create spheres and normal lines for visualization"""
    spheres = []
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Create line set for normals
    normal_lines = o3d.geometry.LineSet()
    line_points = []
    line_indices = []
    
    for i, (point, normal) in enumerate(zip(points, normals)):
        # Create sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)
        sphere.paint_uniform_color([0.9, 0.1, 0.1])  # Red
        spheres.append(sphere)
        
        # Create normal line
        normal_length = 2 * radius
        normal_end = point + normal * normal_length
        
        line_points.append(point)
        line_points.append(normal_end)
        line_indices.append([2*i, 2*i+1])
        
        # Limit to first 5000 points for performance
        if i >= 5000:
            print(f"Warning: Only showing first 5000 normals for performance")
            break
    
    # Add all normal lines at once
    if line_points:
        normal_lines.points = o3d.utility.Vector3dVector(line_points)
        normal_lines.lines = o3d.utility.Vector2iVector(line_indices)
        normal_lines.paint_uniform_color([0.1, 0.1, 0.9])  # Blue
    
    return spheres, normal_lines

def calculate_optimal_sphere_radius(mesh, pcd):
    """Calculate appropriate sphere radius based on mesh scale"""
    mesh_points = np.asarray(mesh.vertices)
    pcd_points = np.asarray(pcd.points)
    
    if len(mesh_points) == 0 or len(pcd_points) == 0:
        return 0.01  # Default value
    
    # Calculate average edge length in mesh
    triangles = np.asarray(mesh.triangles)
    if len(triangles) > 0:
        edge_lengths = []
        for tri in triangles[:1000]:  # Sample first 1000 triangles for speed
            a, b, c = mesh_points[tri]
            edge_lengths.extend([np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)])
        avg_mesh_edge = np.mean(edge_lengths)
    else:
        avg_mesh_edge = np.mean(np.linalg.norm(mesh_points - mesh.get_center(), axis=1))
    
    # Calculate average distance in point cloud
    avg_pcd_dist = np.mean(np.linalg.norm(pcd_points - pcd.get_center(), axis=1))
    
    # Take geometric mean of both scales
    combined_scale = np.sqrt(avg_mesh_edge * avg_pcd_dist)
    
    # Sphere radius should be about 1/20th of the combined scale
    return max(combined_scale / 20, 1e-6)

def process_ply_files_in_directory(
    input_root_dir,
    output_root_dir,
    num_points=500,
    sampling_method='poisson_disk',
    point_noise_mean=0.0,
    point_noise_std=0.0,
    normal_noise_mean=0.0,
    normal_noise_std=0.0,
    overwrite_existing=False
):
    """
    Processes all PLY files in a directory structure, sampling points from meshes and saving results
    with parameter-encoded filenames while preserving folder structure.
    
    Args:
        input_root_dir (str): Root directory containing PLY files
        output_root_dir (str): Root directory for output files
        num_points (int): Number of points to sample
        sampling_method (str): Sampling method ('poisson_disk', 'uniform', 'random')
        point_noise_mean (float): Mean of Gaussian noise for points
        point_noise_std (float): Std of Gaussian noise for points
        normal_noise_mean (float): Mean of Gaussian noise for normals (radians)
        normal_noise_std (float): Std of Gaussian noise for normals (radians)
        overwrite_existing (bool): Whether to overwrite existing output files
        
    Returns:
        dict: Summary of processed files and any errors
    """
    # Create output root directory if it doesn't exist
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'error_messages': [],
        'processed_files': []
    }
    
    # Walk through all subdirectories
    for root, _, files in os.walk(input_root_dir):
        for filename in files:
            if filename.lower().endswith('.ply'):
                input_path = os.path.join(root, filename)
                
                # Create corresponding output directory structure
                relative_path = os.path.relpath(root, input_root_dir)
                output_dir = os.path.join(output_root_dir, relative_path)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Generate parameter-encoded output filename
                output_filename = generate_encoded_filename(
                    filename,
                    num_points,
                    sampling_method,
                    point_noise_mean,
                    point_noise_std,
                    normal_noise_mean,
                    normal_noise_std
                )
                output_path = os.path.join(output_dir, output_filename)
                
                # Skip if file exists and we're not overwriting
                if not overwrite_existing and os.path.exists(output_path):
                    results['skipped'] += 1
                    results['processed_files'].append({
                        'input': input_path,
                        'output': output_path,
                        'status': 'skipped (exists)'
                    })
                    continue
                
                # Process the file
                try:
                    success = sample_points_from_mesh(
                        input_mesh_path=input_path,
                        output_ply_path=output_path,
                        num_points=num_points,
                        sampling_method=sampling_method,
                        point_noise_mean=point_noise_mean,
                        point_noise_std=point_noise_std,
                        normal_noise_mean=normal_noise_mean,
                        normal_noise_std=normal_noise_std
                    )
                    
                    if success:

                        """visualize_with_normals(
                            ply_path1=input_path,
                            ply_path2=output_path,
                            output_image_path=output_path[:-4]+".png",
                            window_name="Original Mesh vs Sampled Points"
                        )"""
                        
                        results['processed'] += 1
                        results['processed_files'].append({
                            'input': input_path,
                            'output': output_path,
                            'status': 'success'
                        })
                    else:
                        results['errors'] += 1
                        results['processed_files'].append({
                            'input': input_path,
                            'output': output_path,
                            'status': 'failed (sampling error)'
                        })
                        
                except Exception as e:
                    results['errors'] += 1
                    error_msg = f"Error processing {input_path}: {str(e)}"
                    results['error_messages'].append(error_msg)
                    results['processed_files'].append({
                        'input': input_path,
                        'output': output_path,
                        'status': f'failed ({str(e)})'
                    })
    
    # Print summary
    print("\nProcessing complete:")
    print(f"- Processed: {results['processed']}")
    print(f"- Skipped (exists): {results['skipped']}")
    print(f"- Errors: {results['errors']}")
    
    if results['errors'] > 0:
        print("\nError messages:")
        for msg in results['error_messages']:
            print(f"  {msg}")
    
    return results

def generate_encoded_filename(
    original_name,
    num_points,
    sampling_method,
    point_noise_mean,
    point_noise_std,
    normal_noise_mean,
    normal_noise_std
):
    """
    Generates a filename that encodes all sampling parameters.
    
    Example: input.ply -> input_p500_poisson_pm0.0_ps0.001_nm0.1_ns0.05.ply
    """
    # Remove .ply extension if present
    base_name = re.sub(r'\.ply$', '', original_name, flags=re.IGNORECASE)
    
    # Create parameter tags
    params = [
        f"p{num_points}",
        sampling_method[:4],  # First 4 chars of method name
        f"ps{point_noise_std:.3f}".replace('.', ''),
        f"ns{normal_noise_std:.3f}".replace('.', '')
    ]
    
    # Join all components
    param_str = '_'.join(params)
    return f"{base_name}_{param_str}.ply"

def main():
    # Example usage
    input_path = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Datasets\erler-2020-p2s-abc\abc\03_meshes\00010218_4769314c71814669ba5d3512_trimesh_013.ply"
    output_path = r"H:\Hands-on_AI_based_3D_Vision\Project_11\sampled_points.ply"
    

    
    num_points_list = [128, 512, 2096]
    noise_levels = [0, 1, 2]
    sampling_methods = ['poisson_disk', 'uniform']

    noise_levels = [0]
    sampling_methods = ['poisson_disk']


    input_root_dir =  r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_2_Dataset"
    output_root_dir = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_2_PointSamples_points"

    for num_points in num_points_list:
        for noise in noise_levels:
            for sampling_method in sampling_methods:
                process_ply_files_in_directory(
                    input_root_dir=input_root_dir,
                    output_root_dir=output_root_dir,
                    num_points=num_points,

                    sampling_method=sampling_method,
                    point_noise_mean=0.0,
                    point_noise_std=0.001 * noise,     # 1mm noise per level
                    normal_noise_mean=0.0,
                    normal_noise_std=0.05 * noise,     # ~2.8 deg per level
                    overwrite_existing=False
                )



    
"""
visualize_with_normals(
    ply_path1=input_path,
    ply_path2=output_path,
    output_image_path=r"H:\Hands-on_AI_based_3D_Vision\Project_11\Sampled_Comparisons__new_test\comparison.png",
    window_name="Original Mesh vs Sampled Points"
)
"""

if __name__ == "__main__":
    main()