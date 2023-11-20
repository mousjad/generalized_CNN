import trimesh
import numpy as np
from pathlib import Path
from tqdm import tqdm

def apply_systematic_deviations(mesh, gradient_x, gradient_y):
    deviated_mesh = mesh.copy()
    deviated_mesh.vertices[:, 0] += gradient_x * mesh.vertices[:, 0] * mesh.vertex_normals[:, 0] / mesh.vertices[:, 0].max()
    deviated_mesh.vertices[:, 1] += gradient_y * mesh.vertices[:, 1] * mesh.vertex_normals[:, 1] / mesh.vertices[:, 1].max()
    return deviated_mesh


def apply_noise(mesh, scale_factor):
    deviated_mesh = mesh.copy()
    random_value = np.random.uniform(-1, 1, mesh.vertices.shape[0]) * scale_factor
    deviated_mesh.vertices += mesh.vertex_normals * random_value.reshape((-1, 1))
    return deviated_mesh

def apply_layerwise_deviations(mesh, num_slices, scale_factor):
    deviated_mesh = mesh.copy()
    z_slices = np.linspace(mesh.bounds[0][2], mesh.bounds[1][2], num_slices + 1)
    for z_start, z_end in zip(z_slices[:-1], z_slices[1:]):
        gradient_x = np.random.uniform(-1, 1) * scale_factor
        gradient_y = np.random.uniform(-1, 1) * scale_factor
        x_phase = np.random.uniform(0, 2 * np.pi)
        y_phase = np.random.uniform(2, 2 * np.pi)
        idx = np.where((mesh.vertices[:, 2] <= z_end) & (mesh.vertices[:, 2] >= z_start))[0]
        deviated_mesh.vertices[idx, 0] += gradient_x * np.sin(2 * np.pi * mesh.vertices[idx, 0] / 100 + x_phase) * \
                                        mesh.vertices[idx, 0] * mesh.vertex_normals[idx,
                                                                                  0] / mesh.vertices[idx, 0].max()
        deviated_mesh.vertices[idx, 1] += gradient_y * np.sin(2 * np.pi * mesh.vertices[idx, 1] / 100 + y_phase) * \
                                        mesh.vertices[idx, 1] * mesh.vertex_normals[idx,
                                                                                  1] / mesh.vertices[idx, 1].max()
    return deviated_mesh


n_sets = 100
set_size = 5
mesh = trimesh.load_mesh("cad_model/mod_nist_light.stl")
path_save = Path("Syn_data")
if not path_save.exists():
    path_save.mkdir()
for i_set in tqdm(range(n_sets)):
    path_set_save = path_save / f"mod_nist_light.{i_set}_set/"
    if not path_set_save.exists():
        path_set_save.mkdir()
    syst_range_x = np.random.uniform(-0.5, 0.5)
    syst_range_y = np.random.uniform(-0.5, 0.5)
    layer_wise = np.random.uniform(0, 0.1)
    for t_set in range(set_size):
        dev_mesh = apply_systematic_deviations(mesh, syst_range_x, syst_range_y)
        dev_mesh = apply_noise(dev_mesh, 0.01)
        dev_mesh = apply_layerwise_deviations(dev_mesh, 70, 0.05)
        dev_mesh.export(path_set_save / f"{t_set}.stl")