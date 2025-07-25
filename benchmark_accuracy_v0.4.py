import open3d as o3d
import numpy as np
import time
from matplotlib import cm
import os
import pandas as pd
import json
import copy

def global_registration(source, target, voxel_size=0.05):
    # Downsample with voxelization
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # Compute FPFH features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    
    # RANSAC registration
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        voxel_size*1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result.transformation

def refine_registration(source, target, transformation, voxel_size=0.05):
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result.transformation

def align_meshes(source_mesh, target_mesh):
    # Convert to point clouds for registration
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=100000)
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=100000)
    
    # Compute normals for both point clouds
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    # 1. Global rough alignment
    transformation = global_registration(source_pcd, target_pcd)
    source_pcd.transform(transformation)
    #print("Global transformation",transformation)
    
    # 2. Fine alignment with ICP
    transformation = refine_registration(source_pcd, target_pcd, transformation)
    source_pcd.transform(transformation)
    #print("Fine alignment transformation",transformation)

    # Apply final transformation to original mesh
    source_mesh.transform(transformation)
    return source_mesh, transformation

def translate_meshes_by_neg_pi_half(mesh):
    """
    Translates (moves) input meshes by -π/2 units in X, Y, and Z directions
    without applying any rotation.
    
    Args:
        *meshes: Variable number of Open3D mesh objects
        
    Returns:
        List of translated meshes
    """
    # Create translation vector [-π/2, -π/2, -π/2]
    translation = -np.pi/2 * np.ones(3)
    
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
        
    mesh_copy.translate(translation)

    
    return mesh_copy

def rotate_mesh(mesh):
    """
    Applies the rotation matrix:
    [ 0  0  1]
    [-1  0  0]
    [ 0 -1  0]
    to a single Open3D TriangleMesh
    
    Args:
        mesh: Open3D TriangleMesh object
    
    Returns:
        Rotated mesh (new object, original is not modified)
    """
    rotation_matrix = np.array([
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0]
    ])
    
    # Create rotated copy
    rotated_mesh = o3d.geometry.TriangleMesh(mesh)
    rotated_mesh = rotated_mesh.rotate(rotation_matrix, center=(0, 0, 0))
    
    return rotated_mesh

def compute_mesh_distance_metrics(original_mesh_path, reconstructed_mesh_path,method, num_points=100000):
    """
    Compute distance metrics between original and reconstructed meshes.
    
    Args:
        original_mesh_path: Path to original mesh file
        reconstructed_mesh_path: Path to reconstructed mesh file
        num_points: Number of points to sample from each mesh
    
    Returns:
        Dictionary containing:
        - chamfer_distance
        - hausdorff_distance
        - sampled_points_original
        - sampled_points_reconstructed
    """
    start_time = time.time()
    #print("Loading meshes...")
    # Load meshes
    mesh_orig = o3d.io.read_triangle_mesh(original_mesh_path)
    mesh_recon = o3d.io.read_triangle_mesh(reconstructed_mesh_path)
    
    mesh_recon = align_to_method(mesh_orig,mesh_recon,method)

    #print(f"Original mesh: {len(mesh_orig.vertices)} vertices")
    #print(f"Reconstructed mesh: {len(mesh_recon.vertices)} vertices")

    # Sample points from both meshes
    pcd_orig = mesh_orig.sample_points_uniformly(number_of_points=num_points)
    pcd_recon = mesh_recon.sample_points_uniformly(number_of_points=num_points)

    # Convert to numpy arrays
    points_orig = np.asarray(pcd_orig.points)
    points_recon = np.asarray(pcd_recon.points)

    # Compute distances using Open3D's built-in functions
    # Chamfer distance
    dist1 = np.asarray(pcd_orig.compute_point_cloud_distance(pcd_recon))
    dist2 = np.asarray(pcd_recon.compute_point_cloud_distance(pcd_orig))
    chamfer_dist = (np.mean(dist1) + np.mean(dist2)) / 2
    
    # Hausdorff distance
    hausdorff_dist = max(dist1.max(), dist2.max())

    metrics = {
        'chamfer_distance': chamfer_dist,
        'hausdorff_distance': hausdorff_dist,
        'sampled_points_original': points_orig,
        'sampled_points_reconstructed': points_recon,
        'num_sampled_points': num_points
    }

    #print("\nDistance Metrics Results:")
    print(f"- Chamfer distance: {metrics['chamfer_distance']:.6f}    - Hausdorff distance: {metrics['hausdorff_distance']:.6f}")
    #print(f"- Hausdorff distance: {metrics['hausdorff_distance']:.6f}")
    end_time = time.time()
    #print(f"Total runtime: {end_time - start_time:.2f} seconds")
    return metrics

# Matplotlib colormap import that works across versions
try:
    # For matplotlib >= 3.7
    from matplotlib.colormaps import get_cmap
except ImportError:
    # For older matplotlib versions
    from matplotlib.cm import get_cmap


def get_white_red_colormap(normalized_values):
    """Generate a white-to-red colormap (white=close, red=far)"""
    # normalized_values: 0 (close) -> 1 (far)
    # White: [1, 1, 1], Red: [1, 0, 0]
    colors = np.zeros((len(normalized_values), 3))
    colors[:, 0] = 1  # Red channel always 1
    colors[:, 1] = 1 - normalized_values  # Green decreases with distance
    colors[:, 2] = 1 - normalized_values  # Blue decreases with distance
    return colors

def get_white_blue_colormap(normalized_values):
    """Generate a white-to-blue colormap (white=close, blue=far)"""
    # normalized_values: 0 (close) -> 1 (far)
    # White: [1, 1, 1], Blue: [0, 0, 1]
    colors = np.zeros((len(normalized_values), 3))
    colors[:, 0] = 1 - normalized_values  # Red decreases with distance
    colors[:, 1] = 1 - normalized_values  # Green decreases with distance
    colors[:, 2] = 1                     # Blue channel always 1
    return colors

def visualize_distance_colormap(original_mesh_path, reconstructed_mesh_path,method, num_points=50000, output_image_path=None,show_window = True):
    try:
        # Load meshes
        mesh_orig = o3d.io.read_triangle_mesh(original_mesh_path)
        mesh_recon = o3d.io.read_triangle_mesh(reconstructed_mesh_path)

        mesh_recon = align_to_method(mesh_orig,mesh_recon,method)

        # Verify meshes
        if not mesh_orig.has_vertices():
            raise ValueError("Original mesh has no vertices")
        if not mesh_recon.has_vertices():
            # Try loading as point cloud if mesh fails
            mesh_recon = o3d.io.read_point_cloud(reconstructed_mesh_path)
            if not mesh_recon.has_points():
                raise ValueError("Reconstructed file is empty (no vertices/points)")

        print(f"Original mesh: {len(mesh_orig.vertices)} vertices "+f"Reconstructed mesh/point cloud: {len(mesh_recon.vertices) if hasattr(mesh_recon, 'vertices') else len(mesh_recon.points)} points" )


        # Create point cloud from original mesh vertices
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = mesh_orig.vertices
        
        # Create point cloud from reconstructed geometry
        if hasattr(mesh_recon, 'sample_points_uniformly'):
            pcd_recon = mesh_recon.sample_points_uniformly(number_of_points=num_points)
        else:
            pcd_recon = mesh_recon  # Already a point cloud

        # Compute distances
        distances = np.asarray(pcd_orig.compute_point_cloud_distance(pcd_recon))
        dist_min, dist_max = distances.min(), distances.max()
        print(f"Distance range: {dist_min:.6f} to {dist_max:.6f}")


        # Normalize distances and apply colormap
        normalized_distances = (distances - dist_min) / (dist_max - dist_min)
        colors = get_white_red_colormap(normalized_distances)
        
        # Apply colors to original mesh
        mesh_orig.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Distance Visualization", width=1200, height=1200)
        vis.add_geometry(mesh_orig)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        

        if show_window:
            vis.run()
        else:
            vis.poll_events()
            vis.update_renderer()
        if output_image_path != "":
            vis.capture_screen_image(output_image_path)
            print(f"Comparison image saved to {output_image_path}")
        vis.destroy_window()
        return True
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise

def align_to_method(mesh_orig,mesh_recon,method):
    if method == "StochasticPSR":
        mesh_recon = translate_meshes_by_neg_pi_half(mesh_recon)
        mesh_recon = rotate_mesh(mesh_recon)
        mesh_recon, transformation = align_meshes(mesh_recon, mesh_orig)
        #print("transformation",transformation)
        print(f"Applied Translation and Rotation")


    if method == "StochasticPSR_wOneSolve":
        mesh_recon = translate_meshes_by_neg_pi_half(mesh_recon)
        print(f"Applied Translation")
    return mesh_recon

def visualize_bidirectional_distance_colormap(original_mesh_path, reconstructed_mesh_path, method,
                                            num_points=50000, output_image_path=None, 
                                            show_window=True, side_by_side=True, 
                                            horizontal_offset=1.5):
    try:
        # Load meshes
        mesh_orig = o3d.io.read_triangle_mesh(original_mesh_path)
        mesh_recon = o3d.io.read_triangle_mesh(reconstructed_mesh_path)
        
        """# Compute centers
        def get_mesh_center(mesh):
            vertices = np.asarray(mesh.vertices)
            return vertices.mean(axis=0)

        center_orig = get_mesh_center(mesh_orig)
        center_recon = get_mesh_center(mesh_recon)

        # Print results
        print("Original mesh center:", center_orig)
        print("Reconstructed mesh center:", center_recon)"""

        mesh_recon = align_to_method(mesh_orig,mesh_recon,method)

        """center_orig = get_mesh_center(mesh_orig)
        center_recon = get_mesh_center(mesh_recon)

        print("After align: Original mesh center:", center_orig)
        print("After align: Reconstructed mesh center:", center_recon)"""


        # Verify meshes
        if not mesh_orig.has_vertices():
            raise ValueError("Original mesh has no vertices")
        if not mesh_recon.has_vertices():
            # Try loading as point cloud if mesh fails
            mesh_recon = o3d.io.read_point_cloud(reconstructed_mesh_path)
            if not mesh_recon.has_points():
                raise ValueError("Reconstructed file is empty (no vertices/points)")

        print(f"Original mesh: {len(mesh_orig.vertices)} vertices")
        print(f"Reconstructed mesh/point cloud: {len(mesh_recon.vertices) if hasattr(mesh_recon, 'vertices') else len(mesh_recon.points)} points")

        # Create point clouds
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = mesh_orig.vertices
        
        if hasattr(mesh_recon, 'sample_points_uniformly'):
            pcd_recon = mesh_recon.sample_points_uniformly(number_of_points=num_points)
        else:
            pcd_recon = mesh_recon  # Already a point cloud

        # Compute distances in both directions
        distances_orig_to_recon = np.asarray(pcd_orig.compute_point_cloud_distance(pcd_recon))
        distances_recon_to_orig = np.asarray(pcd_recon.compute_point_cloud_distance(pcd_orig))

        # Get global min/max for consistent coloring
        global_min = min(distances_orig_to_recon.min(), distances_recon_to_orig.min())
        global_max = max(distances_orig_to_recon.max(), distances_recon_to_orig.max())
        print(f"Global distance range: {global_min:.6f} to {global_max:.6f}")

        # Create colored meshes with different colormaps
        def create_colored_mesh(base_mesh, distances, colormap_fn):
            mesh = copy.deepcopy(base_mesh)
            normalized_distances = (distances - global_min) / (global_max - global_min)
            colors = colormap_fn(normalized_distances)
            
            # CRITICAL FIX: Ensure vertex colors are properly assigned
            if isinstance(mesh, o3d.geometry.TriangleMesh):
                if not mesh.has_vertex_colors():
                    mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(mesh.vertices), 3)))
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            elif isinstance(mesh, o3d.geometry.PointCloud):
                mesh.colors = o3d.utility.Vector3dVector(colors)
            return mesh

        # Create visualizations with different colormaps
        mesh_orig_colored = create_colored_mesh(mesh_orig, distances_orig_to_recon, get_white_red_colormap)
        mesh_recon_colored = create_colored_mesh(mesh_recon, distances_recon_to_orig, get_white_blue_colormap)

        # Offset if side-by-side display
        if side_by_side:
            recon_vertices = np.asarray(mesh_recon_colored.vertices)
            recon_vertices[:, 0] += horizontal_offset  # Offset along x-axis
            mesh_recon_colored.vertices = o3d.utility.Vector3dVector(recon_vertices)

        # Visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Bidirectional Distance Visualization", width=1200, height=800)
        vis.add_geometry(mesh_recon_colored)
        vis.add_geometry(mesh_orig_colored)
        
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True

        if show_window:
            vis.run()
        else:
            vis.poll_events()
            vis.update_renderer()
        
        if output_image_path:
            vis.capture_screen_image(output_image_path)
            print(f"Comparison image saved to {output_image_path}")
        
        vis.destroy_window()
        return True

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise



def create_colorbar(min_val, max_val, height=1.0, width=0.2, position=(1.0, 0.0)):
    """Create a colorbar as a line set"""
    steps = 100
    vertices = []
    lines = []
    colors = []
    
    colormap = get_cmap('viridis')
    
    for i in range(steps):
        y = height * i / steps
        vertices.append([0, y, 0])
        vertices.append([width, y, 0])
        lines.append([2*i, 2*i+1])
        
        # Get color from colormap
        val = min_val + (max_val - min_val) * (i / steps)
        normalized_val = (val - min_val) / (max_val - min_val)
        colors.append(colormap(normalized_val)[:3])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Position the colorbar
    line_set.translate(position)
    return line_set

"""def create_colorbar(min_val, max_val, height=1.0, width=0.2, position=(1.0, 0.0)):
    #Create a colorbar as a line set
    
    steps = 100
    vertices = []
    lines = []
    colors = []
    
    colormap = cm.get_cmap('viridis')
    
    for i in range(steps):
        y = height * i / steps
        vertices.append([0, y, 0])
        vertices.append([width, y, 0])
        lines.append([2*i, 2*i+1])
        
        # Get color from colormap
        val = min_val + (max_val - min_val) * (i / steps)
        normalized_val = (val - min_val) / (max_val - min_val)
        colors.append(colormap(normalized_val)[:3])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Add text labels
    return line_set"""


def batch_evaluate_mesh_distances(
    dataset_mesh_dir,
    recon_base_dir,
    num_points=100000,
    save_path=None
):
    """
    Recursively find all .ply files under dataset_mesh_dir, save their relative folder structure, and for each, find corresponding reconstructions in PSR, SPSR, StochasticPSR, StochasticPSR_wOneSolve.
    Compute mesh distance metrics and store results in a dictionary. Save results to a JSON file.
    """
    import time
    start_time = time.time()
    i = 0
    methods = ["PSR", "SPSR", "StochasticPSR", "StochasticPSR_wOneSolve"]
    results = {}
    print(f"Recursively scanning for .ply files in {dataset_mesh_dir} ...")
    for root, _, files in os.walk(dataset_mesh_dir):
        for fname in files:
            if not fname.endswith(".ply"): continue
            abs_path = os.path.join(root, fname)
            rel_dir = os.path.relpath(root, dataset_mesh_dir)
            rel_path = os.path.join(rel_dir, fname) if rel_dir != '.' else fname
            print(f"\nProcessing: {rel_path}")
            entry = { "original": abs_path, "relative_path": rel_path}
            for method in methods:
                recon_dir = os.path.join(recon_base_dir, method, rel_dir)
                print(f"  Looking for {method} in {recon_dir}")
                
                if not os.path.exists(recon_dir):
                    print(f"    Directory does not exist.")
                    entry[method] = {"error": "Directory not found"}

                    continue
                    
                candidates = [f for f in os.listdir(recon_dir) 
                            if f.startswith(fname[:-4]) and f.endswith(".ply")]
                
                if not candidates:
                    print(f"    No file found for {method}.")
                    entry[method] = {"error": "No matching PLY file found"}
                    continue
                
                entry2 = {}
                for candidate in candidates:
                    recon_path = os.path.join(recon_dir, candidate)
                    print(f"\n method:{method} \tProcessing candidate: {candidate}")
                    
                    try:
                        #print(f"    Computing metrics...")
                        metrics = compute_mesh_distance_metrics(abs_path, recon_path,method, num_points=num_points)
                        
                        current_chamfer = metrics.get("chamfer_distance")
                        current_hausdorff = metrics.get("hausdorff_distance")
                        
                        entry2[recon_path]  = {"chamfer_distance": current_chamfer, "hausdorff_distance": current_hausdorff}
                        
                    except Exception as e:
                        print(f"    Error computing metrics for {candidate}: {e}")
                        entry2[recon_path]  = {"error": str(e),}

                    entry[method] = entry2
            results[rel_path] = entry
            # Save to JSON
            if save_path is None:
                save_path = os.path.join(recon_base_dir, "mesh_distance_results.json")
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {save_path}\nSummary:")
    print(f"\nBatch evaluation complete. Results saved to {save_path}\nSummary:")
    

def make_plots():
    original_mesh = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_2_Dataset\00010218_4769314c71814669ba5d3512_trimesh_013.ply"

    #folder_method_file = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_2_Reconstruction\PSR\00010218_4769314c71814669ba5d3512_trimesh_013_p128_pois_ps0000_ns0000.ply"
    folder = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_2_Reconstruction"
    file = r"00010218_4769314c71814669ba5d3512_trimesh_013_p128_pois_ps0000_ns0000.ply"

    output_image_path = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\ImageComparisons"
    show_window = False

    for method in ["PSR","SPSR","StochasticPSR","StochasticPSR_wOneSolve"]:
        # Example usage
        reconstructed_mesh = os.path.join(folder , method,file)
        visualize_bidirectional_distance_colormap(
            original_mesh_path=original_mesh,
            reconstructed_mesh_path=reconstructed_mesh,
            method = method,
            num_points=50000,
            output_image_path=output_image_path+"\\bidirectional_comparison_"+method+"_offset"+".png",
            show_window=False,
            side_by_side=True
        )
            # Example usage
        visualize_bidirectional_distance_colormap(
            original_mesh_path=original_mesh,
            reconstructed_mesh_path=reconstructed_mesh,
            method = method,
            num_points=50000,
            output_image_path=output_image_path+"\\bidirectional_comparison_"+method+".png",
            show_window= False,
            side_by_side=False
        )

        print("method:",method)
        results = compute_mesh_distance_metrics(
            original_mesh_path=original_mesh,
            reconstructed_mesh_path=reconstructed_mesh,
            method = method,
            num_points=100000
        )

        visualize_distance_colormap(
            original_mesh_path=original_mesh,
            reconstructed_mesh_path=os.path.join(folder , method,file),
            method=method,
            num_points=50000,
            output_image_path = output_image_path+"\original_mesh_distance_to_"+"reconstructed_mesh_"+method+".png",
            show_window = False
        )


# Example usage
if __name__ == "__main__":
    
    dataset_mesh_dir = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Datasets"
    recon_base_dir = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Reconstructions"

    dataset_mesh_dir = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Sampled_Comparisons_Dataset"
    recon_base_dir = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Sampled_Comparisons_Reconstructions"

    dataset_mesh_dir = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_100_Dataset"
    recon_base_dir = r"H:\Hands-on_AI_based_3D_Vision\Project_11\Experiments\Compare_100_Reconstruction"

    batch_evaluate_mesh_distances(
        dataset_mesh_dir,
        recon_base_dir,
        num_points=100000
    )

    #make_plots()
    
