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
os.environ["JAX_PLATFORM_NAME"] = "cpu"    # fall back to CPU

import jax
print(jax.devices())

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


# %%
#currently running with the full path, check after
v, _, _, f, _, _ = igl.readOBJ("/mnt/h/Hands-on_AI_based_3D_Vision/HandsOn_Project_11/GeoSPSR/experiments/meshes/scorpion.obj")
v = normalize_points(v) * np.pi

cam_pos = np.array([[0, -2, 0], [1, 1, 1]]) * np.pi
cam_dir = np.array([[0, 1, 0], [-1, -1, -1]]) * np.pi / 2
p, n = simulate_scan((v), f, cam_pos, cam_dir, np.pi / 3, 200)

rng = default_rng(0)
noise_level = 0.02
p = p + rng.standard_normal(p.shape) * noise_level
n = n + rng.standard_normal(n.shape) * noise_level

# %%
x_grid, f_grid, (x1_grid, x2_grid) = grid_2d_mesh(50)

x_grid = x_grid * np.pi
x_grid = x_grid[:, [0, 2, 1]]
x_grid[:, 1] = -0.75

# %%
plot_mesh_cloud_grid((v, f), (p, n), (x_grid, f_grid))

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

# %%
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

# %%
# Top-Level-Verzeichnis (anpassen je nach Umgebung)
TOP_DIR = os.path.dirname(os.path.abspath(__file__))

# Hilfsfunktion für plattformunabhängige Pfade
def out_path(*args):
    return os.path.join(TOP_DIR, *args)

# Stelle sicher, dass die Ausgabeverzeichnisse existieren
os.makedirs(out_path("output_pngs"), exist_ok=True)
os.makedirs(out_path("output_meshes"), exist_ok=True)

plot_mean(out_path("output_pngs", "cholesky_mean.png"), x1_grid, x2_grid, f_cholesky, save=True)

plot_variance(out_path("output_pngs", "cholesky_variance.png"), x1_grid, x2_grid, var_cholesky, save=True)

# %%
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

# %%
plot_mean(out_path("output_pngs", "sgd_mean.png"), x1_grid, x2_grid, f_sgd, save=True)
plot_variance(out_path("output_pngs", "sgd_variance.png"), x1_grid, x2_grid, var_sgd, save=True)

# %%
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

# %%
for i in range(4):
    plot_mean(out_path("output_pngs", f"sample_{i}.png"), x1_grid, x2_grid, samples[i], save=True)

# %%

# Speichere das generierte Mesh als .ply

vertices = np.array(x_grid)
faces = np.array(f_grid)
vertices_with_height = vertices.copy()
vertices_with_height[:, 2] = np.array(f_cholesky).flatten()

mesh = trimesh.Trimesh(vertices=vertices_with_height, faces=faces)
mesh.export(out_path("output_meshes", "rekonstruierte_oberflaeche.ply"))

# %%
