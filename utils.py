import trimesh
import numpy as np
import open3d as o3d

def measure_distance(mesh, ref_mesh, clean=True):
    import igl
    if type(mesh) == trimesh.base.Trimesh:
        distance, ds, Pos = igl.signed_distance(np.asarray(ref_mesh.vertices), np.asarray(mesh.vertices),
                                            np.asarray(mesh.faces))
    elif type(mesh) == o3d.cpu.pybind.geometry.TriangleMesh:
        distance, ds, Pos = igl.signed_distance(np.asarray(ref_mesh.vertices), np.asarray(mesh.vertices),
                                            np.asarray(mesh.triangles))
    else:
        distance, ds, Pos = igl.signed_distance(np.asarray(ref_mesh.vertices), np.asarray(mesh.vertices),
                                            np.asarray(mesh.faces))
    if clean:
        distance[np.where(np.asarray(ref_mesh.vertices)[:, 2] < 0.03)] = np.zeros_like(
            distance[np.where(np.asarray(ref_mesh.vertices)[:, 2] < 0.03)])
        distance[np.where(np.abs(distance) > 0.35)] = np.zeros_like(
            distance[np.where(np.abs(distance) > 0.35)])
    distance_vec = (distance * np.asarray(ref_mesh.vertex_normals).T).T
    return distance_vec, distance