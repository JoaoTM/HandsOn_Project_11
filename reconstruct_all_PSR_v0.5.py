import os
import numpy as np
import trimesh
from gpytoolbox import point_cloud_to_mesh, stochastic_poisson_surface_reconstruction
from skimage.measure import marching_cubes
from functools import partial
from jax import random, vmap
import jax.numpy as jnp
from numpy.random import default_rng
import json
import time



from geospsr.gp import (
    compute_mean,
    compute_variance,
    f_eigenvalues_from_v_eigenvalues,
    f_gamma_from_v_xi,
    poisson_cross_covariances,
    sample_pathwise_conditioning,
)
from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.utils import periodic_stationary_interpolator, trunc_Zd


def compute_implicit_function_mean(
    x_grid,       # Grid points (N, 3)
    x_data,       # Observed points (M, 3)
    v_data,       # Observed normals (M, 3)
    k_v,          # Vector field kernel
    k_fv,         # Poisson cross-covariance function
    sigma,        # Noise level
    method="sgd", # "sgd" or "cholesky"
    sgd_params=None,
    verbose=True
):
    """
    Compute posterior mean of implicit function f on grid points.
    
    Parameters:
        x_grid: Query points (N,3)
        x_data: Input points (M,3)
        v_data: Input normals (M,3)
        k_v: Vector field kernel
        k_fv: Poisson cross-covariance function  
        sigma: Noise standard deviation
        method: "sgd" (default) or "cholesky"
        sgd_params: Dictionary of SGD parameters (if method="sgd")
        verbose: Print progress messages
        
    Returns:
        Implicit function values at x_grid (N,)
    """
    
    # Default SGD parameters
    default_sgd_params = {
        "key": random.key(0),
        "lr": 1e-3,
        "bs": 128,
        "verbose": verbose,
        "iterations": 1000
    }
    
    if method.lower() == "sgd":
        params = {**default_sgd_params, **(sgd_params or {})}
        data_mean = compute_mean(x_data, x_data, v_data, k_v, k_fv, sigma, 
                               sdd_params=params, verbose=verbose)
        grid_mean = compute_mean(x_grid, x_data, v_data, k_v, k_fv, sigma,
                               sdd_params=params, verbose=verbose)
                               
    elif method.lower() == "cholesky":
        data_mean = compute_mean(x_data, x_data, v_data, k_v, k_fv, sigma,
                               verbose=verbose)
        grid_mean = compute_mean(x_grid, x_data, v_data, k_v, k_fv, sigma,
                               verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sgd' or 'cholesky'")
    
    # Center the implicit function
    return grid_mean - data_mean.mean()

class PointCloudReconstructor:
    def __init__(self, output_base_dir,lengthscale=1e-2, **kwargs):
        """Initialize reconstructor with output directory structure"""
        print(f"Initializing PointCloudReconstructor with output base dir: {output_base_dir}")
        self.output_dirs = {
            'PSR': os.path.join(output_base_dir, "PSR"),
            'SPSR': os.path.join(output_base_dir, "SPSR"),
            'StochasticPSR': os.path.join(output_base_dir, "StochasticPSR"),
            'StochasticPSR_wOneSolve': os.path.join(output_base_dir, "StochasticPSR_wOneSolve")
        }
        self._create_output_dirs()


        box_size=np.pi/2
        lengthscale=1e-2
        variance=0.1
        sigma=0.02

        # Kernel setup
        truncation_n = 50
        amortization_density = 50

        k = Matern32Kernel(lengthscale, variance)
        self.k_v = ProductKernel(k, k, k)

        k_v_eigenvectors = trunc_Zd(truncation_n)
        k_v_eigenvalues = vmap(self.k_v.spectral_density)(k_v_eigenvectors) ** 0.5

        k_fv_expensive = partial(
            poisson_cross_covariances,
            eigenvectors=k_v_eigenvectors,
            eigenvalues=k_v_eigenvalues,
        )
        self.k_fv = periodic_stationary_interpolator(
            k_fv_expensive, 3, amortization_density, exponent=5, verbose=True
        )


    
        
    def _create_output_dirs(self):
        """Ensure all output directories exist"""
        print("Creating output directories...")
        for method, dir_path in self.output_dirs.items():
            print(f"  Creating directory for {method}: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        
    
    def turn_point_cloud_to_mesh(self, P, N, method='PSR', **kwargs):
        """
        Convert point cloud to mesh using specified method.
        """
        print(f"Input points shape: {P.shape}, normals shape: {N.shape}")
        
        method = method.lower()
        if method == 'psr':
            return self._psr_reconstruction(P, N)
        elif method == 'spsr':
            return self._spsr_reconstruction(P, N)
        elif method == 'stochasticpsr':
            return self._stochastic_psr_reconstruction(P, N, **kwargs)
        elif method == 'stochasticpsr_wonesolve':
            return self._stochastic_psr_onesolve(P, N, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _psr_reconstruction(self, P, N, 
                          screening_weight=1.0, 
                          depth=10, 
                          outer_boundary_type='Neumann',
                          verbose=False):
        """Standard Poisson Surface Reconstruction"""
        print(f"Running PSR with screening_weight={screening_weight}, depth={depth}")
        print(f"First point: {P[0]}, First normal: {N[0]}")
        
        V, F = point_cloud_to_mesh(
            P=P,
            N=N,
            psr_screening_weight=screening_weight,
            psr_depth=depth,
            psr_outer_boundary_type=outer_boundary_type,
            verbose=verbose
        )
        print(f"PSR completed. Output vertices: {V.shape}, faces: {F.shape}")
        return V, F

    def _spsr_reconstruction(self, P, N, 
                           screening_weight=4.0,
                           depth=10,
                           outer_boundary_type='Neumann',
                           verbose=False, **kwargs):
        """Screened Poisson Surface Reconstruction"""
        print(f"Running SPSR with screening_weight={screening_weight}, depth={depth}")
        
        V, F = point_cloud_to_mesh(
            P=P,
            N=N,
            psr_screening_weight=screening_weight,
            psr_depth=depth,
            psr_outer_boundary_type=outer_boundary_type,
            verbose=verbose
        )
        print(f"SPSR completed. Output vertices: {V.shape}, faces: {F.shape}")
        return V, F

    def _stochastic_psr_reconstruction(self, P, N, 
                                 grid_res=80, 
                                 box_size=np.pi/2,
                                 sigma=0.05, **kwargs):
        """Stochastic Poisson Surface Reconstruction"""
        print(f"Running StochasticPSR with grid_res={grid_res}, box_size={box_size}")
        
        gs = np.array([grid_res]*3, dtype=np.int32)
        h_val = 2*box_size/(grid_res-1)
        h = np.array([h_val]*3, dtype=np.float32)
        corner = np.array([-box_size]*3, dtype=np.float32)
        
        print("Calling stochastic_poisson_surface_reconstruction...")
        result = stochastic_poisson_surface_reconstruction(
            P=np.array(P, dtype=np.float32),
            N=np.array(N, dtype=np.float32),
            gs=gs,
            h=h,
            corner=corner,
            sigma=sigma
        )
        
        # Handle different return types from gpytoolbox
        if isinstance(result, tuple):
            scalar_mean = result[0]  # First element is the scalar field
        else:
            scalar_mean = result
            
        print("Stochastic reconstruction completed. Running marching cubes...")
        
        # Reshape scalar field to 3D grid
        try:
            scalar_field = scalar_mean.reshape(gs)
        except AttributeError:
            # If it's already a numpy array but can't be reshaped
            scalar_field = np.array(scalar_mean).reshape(gs)
        
        vertices, faces, _, _ = marching_cubes(
            scalar_field,
            level=0,
            spacing=[h_val]*3
        )
        print(f"Marching cubes completed. Vertices: {vertices.shape}, faces: {faces.shape}")
        return vertices, faces

    def _stochastic_psr_onesolve(self, P, N, 
                            grid_res=80,
                            box_size=np.pi/2,
                            lengthscale=1e-2,
                            variance=0.1,
                            sigma=0.02):
        """Stochastic PSR with one-time solve approach"""
        print(f"Running StochasticPSR_wOneSolve with grid_res={grid_res}")
        
        # Convert inputs to JAX arrays explicitly
        P_jax = jnp.array(P, dtype=jnp.float32)
        N_jax = jnp.array(N, dtype=jnp.float32)
        
        """# Kernel setup
        truncation_n = 50
        amortization_density = 50

        k = Matern32Kernel(lengthscale, variance)
        k_v = ProductKernel(k, k, k)

        k_v_eigenvectors = trunc_Zd(truncation_n)
        k_v_eigenvalues = vmap(k_v.spectral_density)(k_v_eigenvectors) ** 0.5

        k_fv_expensive = partial(
            poisson_cross_covariances,
            eigenvectors=k_v_eigenvectors,
            eigenvalues=k_v_eigenvalues,
        )
        k_fv = periodic_stationary_interpolator(
            k_fv_expensive, 3, amortization_density, exponent=5, verbose=True
        )"""
        # Prepare grid as JAX array
        grid_3d = np.mgrid[
            -box_size:box_size:grid_res*1j, 
            -box_size:box_size:grid_res*1j, 
            -box_size:box_size:grid_res*1j
        ].reshape(3, -1).T
        grid_3d_jax = jnp.array(grid_3d, dtype=jnp.float32)
        print(f"Grid prepared with shape: {grid_3d.shape}")
        
        print("Computing implicit function...")
        try:
            # Ensure we're working with JAX arrays
            f_mean = compute_implicit_function_mean(
                grid_3d_jax, P_jax, N_jax, self.k_v, self.k_fv, sigma
            )
            
            # Convert back to numpy for marching cubes
            f_mean_np = np.array(f_mean)
            print(f"Implicit function computed. Shape: {f_mean_np.shape}")
        except Exception as e:
            print(f"Failed to compute implicit function: {e}")
            raise
        
        print("Running marching cubes...")
        try:
            # Reshape and ensure contiguous array
            scalar_field = np.ascontiguousarray(f_mean_np.reshape((grid_res, grid_res, grid_res)))
            spacing = [2*box_size/(grid_res-1)]*3
            
            vertices, faces, _, _ = marching_cubes(
                scalar_field,
                level=0,
                spacing=spacing
            )
            print(f"Marching cubes completed. Vertices: {vertices.shape}, faces: {faces.shape}")
            return vertices, faces
        except Exception as e:
            print(f"Failed in marching cubes: {e}")
            raise


    def save_mesh(self, V, F, method, filename):
        """Save reconstructed mesh to appropriate directory"""
        print(f"\nSaving {method} result for {filename}")
        print(f"Vertices: {V.shape}, Faces: {F.shape}")
        
        # Create the full output path while preserving the original subdirectory structure
        output_path = os.path.join(self.output_dirs[method], filename)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Output path: {output_path}")
        
        try:
            mesh = trimesh.Trimesh(V, F)
            mesh.export(output_path)
            print("Mesh saved successfully")
            return output_path
        except Exception as e:
            print(f"Failed to save mesh: {e}")
            raise

def read_ply_with_normals(file_path):
    """Manually read PLY file containing points and normals"""
    points = []
    normals = []
    with open(file_path, 'r') as f:
        # Read header
        vertex_count = 0
        has_normals = False
        while True:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property') and 'nx' in line:
                has_normals = True
            elif line.startswith('end_header'):
                break
        
        # Read data
        for _ in range(vertex_count):
            line = f.readline().strip()
            if not line:
                continue
            values = list(map(float, line.split()))
            points.append(values[:3])
            if has_normals:
                normals.append(values[3:6])
    
    points = np.array(points)
    normals = np.array(normals) if has_normals else np.ones_like(points)
    return points, normals

def batch_process_directory(input_dir, output_base_dir, **kwargs):
    """Process all PLY files in directory with all methods and save timings"""
    # Convert paths to current OS format
    input_dir = os.path.normpath(input_dir.replace('H:', '/mnt/h'))
    output_base_dir = os.path.normpath(output_base_dir.replace('H:', '/mnt/h'))
    
    print(f"\nStarting batch processing of directory: {input_dir}")
    print(f"Output will be saved to: {output_base_dir}")
    
    reconstructor = PointCloudReconstructor(output_base_dir, **kwargs)
    timing_data = []
    print(f"test2")

    
    file_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.ply'):
                file_count += 1
                input_path = os.path.normpath(os.path.join(root, file))
                print(f"\n\nProcessing file {file_count}: {input_path}")
                
                try:
                    # Load point cloud using our custom reader
                    P, N = read_ply_with_normals(input_path)
                    print(f"Loaded {len(P)} points with normals")
                    
                    # Process with all methods
                    for method in ['PSR', 'SPSR', 'StochasticPSR', 'StochasticPSR_wOneSolve']:
                        print(f"\nAttempting {method} reconstruction...")
                        try:
                            # Create relative path for output
                            rel_path = os.path.relpath(root, start=input_dir)
                            output_filename = os.path.join(rel_path, file) if rel_path != '.' else file
                            output_path = os.path.join(output_base_dir,method, output_filename)


                            if os.path.exists(output_path):
                                print("Output Path exists:",output_path)
                                continue
                            
                            start_time = time.time()
                            V, F = reconstructor.turn_point_cloud_to_mesh(P, N, method=method, **kwargs)
                            elapsed = time.time() - start_time
                            print(f"Time for {method}: {elapsed:.3f} seconds")
                            
                            # Record timing data
                            timing_data.append({
                                'input_path': input_path,
                                'method': method,
                                'time': elapsed,
                                'status': 'success'
                            })
                            
                            
                            output_path = reconstructor.save_mesh(V, F, method, output_filename)
                            print(f"SUCCESS: {method} -> {output_path}")
                            
                        except Exception as e:
                            error_msg = str(e)
                            print(f"FAILED {method}: {error_msg}")
                            timing_data.append({
                                'input_path': input_path,
                                'method': method,
                                'time': None,
                                'status': 'failed',
                                'error': error_msg
                            })
                            
                except Exception as e:
                    print(f"Failed to process {file}: {str(e)}")
                    timing_data.append({
                        'input_path': input_path,
                        'method': 'all',
                        'time': None,
                        'status': 'failed',
                        'error': str(e)
                    })

                # Save timing data to JSON
                times_json_path = os.path.join(output_base_dir, 'times.json')
                print("saving time data at :",times_json_path)
                with open(times_json_path, 'w') as f:
                    json.dump(timing_data, f, indent=2)
    
    print(f"\nBatch processing complete. Processed {file_count} files.")

if __name__ == "__main__":
    print("Starting point cloud reconstruction pipeline...")
    
    # Example Linux-compatible paths
    input_dir = "/mnt/h/Hands-on_AI_based_3D_Vision/Project_11/Sampled_Points"
    output_base_dir = "/mnt/h/Hands-on_AI_based_3D_Vision/Project_11/Reconstructions"


    #Steppermotor
    input_dir = "/mnt/h/Hands-on_AI_based_3D_Vision/Project_11/Experiments/Compare_100_PointSamples"
    output_base_dir = "/mnt/h/Hands-on_AI_based_3D_Vision/Project_11/Experiments/Compare_100_Reconstruction"

    """input_dir = "/mnt/h/Hands-on_AI_based_3D_Vision/Project_11/Experiments/Compare_2_PointSamples_points"
    output_base_dir = "/mnt/h/Hands-on_AI_based_3D_Vision/Project_11/Experiments/Compare_2_Reconstruction_amortization_density80_truncation_n60"
    """

    batch_process_directory(input_dir, output_base_dir)
    print("Pipeline execution completed.")