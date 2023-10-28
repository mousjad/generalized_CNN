import trimesh
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from git import Repo

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
        distance[np.where(np.abs(distance) > 0.5)] = np.zeros_like(
            distance[np.where(np.abs(distance) > 0.5)])
    distance_vec = (distance * np.asarray(ref_mesh.vertex_normals).T).T
    return distance_vec, distance

def view3d(ref_mesh, cm):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D scatter with colormap based on z-coordinate
    sc = ax.scatter(ref_mesh.vertices[:,0], ref_mesh.vertices[:,1], ref_mesh.vertices[:,2], c=cm, cmap='jet')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add colorbar for the colormap
    cbar = fig.colorbar(sc)
    cbar.set_label('Color Map')

    # Show the plot
    plt.show()

def git_push(path, commit):
    try:
        repo = Repo(path)
        repo.git.add(update=True)
        repo.index.commit(commit)
        origin = repo.remote(name='origin')
        origin.push()
    except Exception as error:
        print(f'Some error occured while pushing the code.\n{error}')