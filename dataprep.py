import numpy as np
import trimesh
import pickle
from tqdm import tqdm
import igl
import os
from utils import measure_distance
from copy import deepcopy
import multiprocessing as mp
import torch
from itertools import combinations
from multiprocessing import Pool, cpu_count
import logging
logging.getLogger("trimesh").setLevel(logging.ERROR)

def create_scan_dist(dir_id="scan_data/"):

    master_scan_dist_list = []
    master_ave_dist_list = []
    master_ref_mesh_list = []

    for subdir_id in tqdm(os.listdir(dir_id)):
        ref_mesh_fid = "cad_model/" + subdir_id.split('.')[0] + ".stl"
        ref_mesh = trimesh.load(ref_mesh_fid)
        subdir_id = dir_id + subdir_id + "/"


        l_dir = os.listdir(subdir_id)
        combs = combinations(l_dir, 4)
        for comb in combs:
            l_dist = []
            for f_id in comb:
                f_id = subdir_id + f_id
                scan_mesh = trimesh.load(f_id)
                _, dist = measure_distance(scan_mesh, ref_mesh)
                master_scan_dist_list.append(dist)
                master_ref_mesh_list.append(ref_mesh_fid)

                l_dist.append(dist)
            ave_dist = np.array(l_dist).mean(axis=0)
            for i in comb:
                master_ave_dist_list.append(ave_dist)

    return master_scan_dist_list, master_ave_dist_list, master_ref_mesh_list

def process_mesh(args):
    i, fid, master_scan_dist_list = args
    for i in range(master_scan_dist_list.__len__()):
        p_fid = 'cad_indices/' + fid[i].split('/')[1].split('.')[0] + '.pkl'
        if fid[i].split('/')[1].split('.')[0] + '.pkl' not in os.listdir("cad_indices"):
            p = create_conv_image_indices(trimesh.load(fid[i]), 15, 0.75, p_fid)
        else:
            with open(p_fid, 'rb') as f:
                p = pickle.load(f)


        if i==0:
            conv = create_conv_image_from_indices(p, master_scan_dist_list[i], 15, False)
        else:
            conv = torch.cat((conv,
                              create_conv_image_from_indices(p, master_scan_dist_list[i], 15, False)), axis=0)
    return conv

def combine_results(results):
    for i in range(results.__len__()):
        if i == 0:
            master_conv = results[i].reshape(-1, 15, 15)
        else:
            master_conv = torch.cat((master_conv, results[i].reshape(-1, 15, 15)), axis=0)
    return master_conv


def create_conv_data(master_scan_dist_list, master_ref_mesh_list):
    num_processes = cpu_count()  # Number of CPU cores
    data_length = len(master_ref_mesh_list)
    chunk_size = data_length // num_processes

    # Split the data into chunks for each process
    chunks = [(i, master_ref_mesh_list[i*chunk_size:np.min(((i+1)*chunk_size, master_ref_mesh_list.__len__()))],
               master_scan_dist_list[i*chunk_size:np.min(((i+1)*chunk_size, master_ref_mesh_list.__len__()))])
              for i in range(num_processes+1)]

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_mesh, chunks), total=len(chunks)))

    # Combine the results from different processes
    master_conv = combine_results(results)

    return master_conv

def create_conv_image_indices(ref_mesh, shape=15, step=0.75, f_id = 'p.pkl'):
    # Calculate sharp edges and other values
    k_tree = ref_mesh.kdtree
    SE, _, _, _, _, _ = igl.sharp_edges(ref_mesh.vertices, ref_mesh.faces, 1.4)

    # Divide vertices into chunks for parallel processing
    num_vertices = ref_mesh.vertices.shape[0]
    chunk_size = num_vertices // mp.cpu_count()
    vertex_chunks = [
        ref_mesh.vertices[i:i + chunk_size, :]
        for i in range(0, num_vertices, chunk_size)
    ]

    print('Starting pool of processes')
    # Create a pool of processes
    with mp.Pool() as pool:
        # Map the compute_indices function over the vertex chunks in parallel, passing
        # the necessary arguments to the function as a tuple
        results = pool.starmap(
            compute_indices,
            [(chunk, ref_mesh, shape, step, SE, k_tree) for chunk in vertex_chunks]
        )

    # Combine the results from each process
    all_scan_case_dist = [item for sublist in results for item in sublist]

    # Save the results to a file and return them
    with open(f_id, 'wb') as f:
        pickle.dump(all_scan_case_dist, f)
        print('saved')
    return all_scan_case_dist

def compute_indices(vertices, ref_mesh, shape, step, SE, k_tree):
    all_scan_case_dist = []

    # Compute indices for vertices in chunk
    for i, vert in enumerate(vertices):
        ni = ref_mesh.vertex_normals[i, :]

        # Creation of the y axis of the conv cases, either toward the center of the mesh or the top
        if np.abs(ni).argmax() == 2:
            ori = ref_mesh.vertices[i, :] - np.array((0, 0, ref_mesh.vertices[i, 2]))
            if np.all(ori == np.zeros(3)):
                ori = np.array((1, 0, 0))
            else:
                ori = ori / np.linalg.norm(ori)
        else:
            ori = np.array((0, 0, 1))

        peri = np.cross(ni, ori)
        ori = np.cross(peri, ni)

        X, Y = [], []
        for s in np.linspace(-(shape - 1) / 2 * step, (shape - 1) / 2 * step, shape):
            X.append(peri * s)
            Y.append(ori * s)
        X = np.array(X)
        Y = np.array(Y)

        scan_case_dist = []
        for u1, x in enumerate(X):
            for v1, y in enumerate(Y):
                v = ref_mesh.vertices[i, :] - x - y
                p = k_tree.query_ball_point(v, np.sqrt(2 * (step) ** 2))
                if len(p) != 0:
                    # p = list(set(p) - set(list(np.unique(SE))))
                    p = np.asarray(p)[
                        np.nonzero(ref_mesh.vertex_normals[p, :].dot(ref_mesh.vertex_normals[i, :]) > 0.67)]
                if len(p) == 0:
                    scan_case_dist.append(-1)
                else:
                    scan_case_dist.append(p)
        all_scan_case_dist.append(scan_case_dist)

    # Return list of computed indices for all vertices in the chunk
    return all_scan_case_dist

def create_conv_image_from_indices(indices, scan_dist, shape=15, show_p_bar=True):
    l_scan_case_dist = []
    scan_case_dist = np.ones((shape, shape))
    if show_p_bar:
        for u, ind in enumerate(tqdm(indices)):
            for i, p in enumerate(ind):
                if np.any(p == -1):
                    scan_case_dist[i % shape, i // shape] = 0
                else:
                    scan_case_dist[i % shape, i // shape] = scan_dist[p].mean() - scan_dist[u]
            l_scan_case_dist.append(scan_case_dist.copy())
    else:
        for u, ind in enumerate(indices):
            for i, p in enumerate(ind):
                if np.any(p == -1):
                    scan_case_dist[i % shape, i // shape] = 0
                else:
                    scan_case_dist[i % shape, i // shape] = scan_dist[p].mean() - scan_dist[u]
            l_scan_case_dist.append(scan_case_dist.copy())
    return torch.tensor(np.array(l_scan_case_dist))


if __name__ == '__main__':
    # master_scan_dist_list, master_ave_dist_list, master_ref_mesh_list = create_scan_dist()
    #
    # with open('temp/master_scan_dist_list.pkl', 'wb') as f:
    #     pickle.dump(master_scan_dist_list, f)
    # with open('temp/master_ave_dist_list.pkl', 'wb') as f:
    #     pickle.dump(master_ave_dist_list, f)
    # with open('temp/master_ref_mesh_list.pkl', 'wb') as f:
    #     pickle.dump(master_ref_mesh_list, f)


    with open('temp/master_scan_dist_list.pkl', 'rb') as f:
        master_scan_dist_list = pickle.load(f)
    with open('temp/master_ave_dist_list.pkl', 'rb') as f:
        master_ave_dist_list = pickle.load(f)
    with open('temp/master_ref_mesh_list.pkl', 'rb') as f:
        master_ref_mesh_list = pickle.load(f)

    master_conv = create_conv_data(master_scan_dist_list, master_ref_mesh_list)
    torch.save(master_conv, "data/master_conv.trc")
    print('done')
