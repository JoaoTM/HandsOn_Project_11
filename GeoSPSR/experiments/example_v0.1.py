# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
#os.environ["JAX_PLATFORM_NAME"] = "cpu"    # fall back to CPU

import jax
print(jax.devices())

from gpytoolbox import stochastic_poisson_surface_reconstruction  
from functools import partial
import igl
import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from numpy.random import default_rng
from utils import (
    grid_2d_mesh,
    normalize_points,
    plot_mean,
    plot_mesh_cloud_grid,
    plot_variance,
    simulate_scan,
)

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

import trimesh
from skimage.measure import marching_cubes
from meshplot import plot as plot_mesh


# Top-Level-Verzeichnis (anpassen je nach Umgebung)
TOP_DIR = os.path.dirname(os.path.abspath(__file__))

# Hilfsfunktion für plattformunabhängige Pfade
def out_path(*args):
    return os.path.join(TOP_DIR, *args)

# Stelle sicher, dass die Ausgabeverzeichnisse existieren
os.makedirs(out_path("output_pngs"), exist_ok=True)
os.makedirs(out_path("output_meshes"), exist_ok=True)


# %%
#currently running with the full path, check after
v, _, _, f, _, _ = igl.readOBJ("/mnt/h/Hands-on_AI_based_3D_Vision/HandsOn_Project_11/GeoSPSR/experiments/meshes/scorpion.obj")
v = normalize_points(v) * np.pi

cam_pos = np.array([[0, -2, 0], [1, 1, 1]]) * np.pi             #   virtual camera position 
cam_dir = np.array([[0, 1, 0], [-1, -1, -1]]) * np.pi / 2      #   virtual camera direction
p, n = simulate_scan((v), f, cam_pos, cam_dir, np.pi / 3, 200)

rng = default_rng(0)
noise_level = 0.02
p = p + rng.standard_normal(p.shape) * noise_level
n = n + rng.standard_normal(n.shape) * noise_level

# %%
x_grid, f_grid, (x1_grid, x2_grid) = grid_2d_mesh(100)

# ... existing code ...

# Nach der Erstellung von x_grid, f_grid, x1_grid, x2_grid
print(f"Shape of x_grid: {x_grid.shape}")
print(f"x_grid\n: {x_grid}")
print(f"Shape of f_grid: {f_grid.shape}")
print(f"Shape of x1_grid: {x1_grid.shape}")
print(f"Shape of x2_grid: {x2_grid.shape}")



# ... existing code ...

x_grid = x_grid * np.pi
x_grid = x_grid[:, [0, 2, 1]]
x_grid[:, 1] = -0.75

# %%
meshPlot =plot_mesh_cloud_grid((v, f), (p, n), (x_grid, f_grid))
#meshPlot.savefig(out_path("output_pngs", "plot_mesh_cloud_grid.png", bbox_inches="tight"), pad_inches=0, dpi=400)

meshPlot.save(out_path("output_pngs", "plot_mesh_cloud_grid"))


# %%
sigma = noise_level
lengthscale = 1e-2
variance = 0.1
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
)

k_f_eigenvalues = f_eigenvalues_from_v_eigenvalues(k_v_eigenvectors, k_v_eigenvalues)
k_f_variance = (k_f_eigenvalues**2).sum()

# %%
x_data = jnp.array(p)
v_data = jnp.array(n)

# Nach der Erstellung von x_data und v_data
print(f"Shape of x_data: {x_data.shape}")
print(f"Shape of v_data: {v_data.shape}")


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


def extract_mesh_SPSR_wOneSolve(
    x_grid, x_data, v_data, k_v, k_fv, sigma,
    grid_res, output_path="output.ply"
    ):

    """Full pipeline from GP to mesh."""
    # 1. Compute implicit function mean
    f_mean = compute_implicit_function_mean(
        x_grid, x_data, v_data, k_v, k_fv, sigma
    )
    
    # 2. Reshape to 3D grid (assuming x_grid is ordered as a grid)
    f_grid = f_mean.reshape((grid_res, grid_res, grid_res))
    
    # Calculate spacing based on the actual range of x_grid
    min_vals = np.min(x_grid, axis=0)
    max_vals = np.max(x_grid, axis=0)
    ranges = max_vals - min_vals
    spacing = ranges / (grid_res - 1)
    print(f"spacing\n: {spacing}")

    # 4. Run marching cubes
    vertices, faces, _, _ = marching_cubes(
        np.array(f_grid),  # Convert JAX array to NumPy
        level=0,
        spacing=spacing
    )
    
    # 5. Save as PLY
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(output_path)
    print(f"Mesh saved to {output_path}")
    return mesh

def extract_mesh_SPSR(
    P, N, grid_res=80, box_size=np.pi/2, sigma=0.05, 
    output_path="output.ply", verbose=True
):
    """Wrapper that matches your wOneSolve interface"""
    # Convert grid parameters to numpy arrays
    gs = np.array([grid_res]*3, dtype=np.int32)
    h_val = 2*box_size/(grid_res-1)
    h = np.array([h_val]*3, dtype=np.float32)
    corner = np.array([-box_size]*3, dtype=np.float32)
    
    # Run reconstruction
    result = stochastic_poisson_surface_reconstruction(
        P=np.array(P, dtype=np.float32),
        N=np.array(N, dtype=np.float32),
        gs=gs,
        h=h,
        corner=corner,
        sigma=sigma,
        verbose=verbose
    )
    
    scalar_mean = result[0]
    
    # Reshape scalar field to 3D
    try:
        scalar_mean_3d = scalar_mean.reshape(gs)
    except ValueError as e:
        raise ValueError(f"Could not reshape {scalar_mean.shape} to {gs}") from e
    
    # Use uniform spacing (h_val in each dimension)
    spacing = (h_val, h_val, h_val)  # Tuple of scalars
    
    # Extract isosurface
    vertices, faces, _, _ = marching_cubes(
        scalar_mean_3d,
        level=0,
        spacing=spacing  # Now correctly formatted
    )
    
    # Save mesh
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(output_path)
    if verbose:
        print(f"Mesh saved to {output_path}")
    return mesh

# Erstelle ein 3D-Grid mit 20 Werten pro Achse im Bereich von -0.5 bis 0.5
grid_res = 80
box_size = np.pi /2        # (*2)
grid_3D = np.mgrid[-box_size:box_size:grid_res*1j, -box_size:box_size:grid_res*1j, -box_size:box_size:grid_res*1j]
grid_3D = grid_3D.reshape(3, -1).T
print(f"Shape of grid_3D: {grid_3D.shape}")
print(f"grid_3D\n: {grid_3D}")


# Same calling convention as wOneSolve version
extract_mesh_SPSR(
    P=x_data,
    N=v_data,
    grid_res=80,         # Single integer
    box_size=np.pi/2,    # Same as before
    output_path=out_path("output_meshes", "SPSR_reconstructed_surface.ply"),
)


extract_mesh_SPSR_wOneSolve(
    grid_3D, x_data, v_data, k_v, k_fv, sigma,
    grid_res,  # Must match your x_grid generation
    output_path=out_path("output_meshes", "gp_SPSR_wOneSolve_reconstructed_surface.ply")
)




