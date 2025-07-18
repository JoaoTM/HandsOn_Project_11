#Test to run a dataset
# %%
#Imports from original code

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

#%%
#My imports
import os
import open3d as o3d

#%%
#Basic functions

def load_ply(path):
    mesh = o3d.io.read_triangle_mesh(path)
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return points,faces,mesh

def load_npy(path):
    points = np.load(path)
    return points

def array2pcld(points:np.array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float32))
    #in case of colors
    if points.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6].astype(np.float32))
    return pcd

def get_normals(pc_array:np.array):
    pcd = array2pcld(pc_array)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1,  # Search radius
        max_nn=30    # Max neighbors to consider
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    normals = np.asanyarray(pcd.normals)
    return normals

# %%
#Full functions
def run_file(path,vals):
    """
    Assuming we are running the pointclouds
    """
    p = normalize_points(load_npy(path))*np.pi
    n = get_normals(p)
    #missing is the f_grid, it's not used
    x_grid, _ , (x1_grid, x2_grid) = grid_2d_mesh(100)

    x_grid = x_grid * np.pi
    x_grid = x_grid[:, [0, 2, 1]]
    x_grid[:, 1] = -0.75

    sigma = vals[0]
    lengthscale = vals[1]
    variance = vals[2]
    truncation_n = vals[3]
    amortization_density = vals[4]

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

    x_data = jnp.array(p)
    v_data = jnp.array(n)
    
    data_mean_cholesky = compute_mean(
    x_data, x_data, v_data, k_v, k_fv, sigma, verbose=True
    )
    grid_mean_cholesky = compute_mean(
        x_grid, x_data, v_data, k_v, k_fv, sigma, verbose=True
    )
    f_cholesky = grid_mean_cholesky - data_mean_cholesky.mean()

    var_cholesky = compute_variance(
        x_grid, x_data, k_v, k_fv, k_f_variance, sigma, verbose=True
    )

    sdd_params = {
    "key": random.key(0),
    "lr": 1e-3,
    "bs": 128,
    "verbose": True,
    "iterations": 1000,
    }
    data_mean_sgd = compute_mean(
        x_data, x_data, v_data, k_v, k_fv, sigma, sdd_params=sdd_params, verbose=True
    )
    grid_mean_sgd = compute_mean(
        x_grid, x_data, v_data, k_v, k_fv, sigma, sdd_params=sdd_params, verbose=True
    )

    f_sgd = grid_mean_sgd - data_mean_sgd.mean()

    var_sgd = compute_variance(
        x_grid, x_data, k_v, k_fv, k_f_variance, sigma, sdd_params=sdd_params, verbose=True
    )

    xi = random.normal(random.key(0), shape=(4, 3, k_v_eigenvectors.shape[0], 2))
    gamma = f_gamma_from_v_xi(xi, k_v_eigenvectors, k_v_eigenvalues)

    samples = sample_pathwise_conditioning(
        x_grid,
        x_data,
        v_data,
        k_v,
        k_fv,
        sigma,
        xi,
        gamma,
        k_v_eigenvectors,
        k_v_eigenvalues,
        k_v_eigenvectors,
        k_f_eigenvalues,
        2**6,
        2**6,
        2**6,
        verbose=True,
    )
    #for i in range(4):
    #    plot_mean("", x1_grid, x2_grid, samples[i], save=False)
    return data_mean_cholesky,var_cholesky,data_mean_sgd,var_sgd


# %%
vals = [0.2,1e-2,0.1,50,50]
dataset_dir = r"C:\Users\joaot\Desktop\Universidade\4oAno2oSemestre\Hands-On_AI_3D_Vision\Project_11\ABCdataset\abc\04_pts"

# %%

if __name__ == "__main__":
    for name in os.listdir(dataset_dir):
        dir = os.path.join(dataset_dir,name)
        dmc,vc,dms,vs = run_file(dir,vals)
        print(f"Printing Cholesky data from {name}")
        print(dmc.mean())
        print(vc.mean())
        print("......................................")
        print(f"Printing SGD data from {name}")
        print(dms.mean())
        print(vs.mean())

