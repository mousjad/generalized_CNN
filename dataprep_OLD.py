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

def create_scan_dist(dir_id="scan_data/", mode='train'):

    master_scan_dist_list = []
    master_ave_dist_list = []
    master_ref_mesh_list = []
    master_scan_dist_light_list = []
    fid_dict = {'test': '_TEST', 'train': 'TRAIN'}

    for subdir_id in tqdm(os.listdir(dir_id)):
        if subdir_id[-5:] == fid_dict[mode] and subdir_id[0] != ".":
            ref_mesh_fid = "cad_model/" + subdir_id.split('.')[0] + ".stl"
            ref_mesh_ligh_fid = "cad_model/" + subdir_id.split('.')[0] + "_light.stl"
            ref_mesh = trimesh.load(ref_mesh_fid)
            ref_mesh_light = trimesh.load_mesh(ref_mesh_ligh_fid)
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
                    _, dist_light = measure_distance(scan_mesh, ref_mesh_light)
                    master_scan_dist_light_list.append(dist_light)
                    master_ref_mesh_list.append(ref_mesh_fid)

                    l_dist.append(dist_light)
                ave_dist = np.array(l_dist).mean(axis=0)
                for i in comb:
                    master_ave_dist_list.append(ave_dist)
                # print(master_ave_dist_list.__len__(), (ave_dist - l_dist).std(axis=1))

    return master_scan_dist_list, master_ave_dist_list, master_ref_mesh_list, master_scan_dist_light_list

def process_mesh(args):
    fid, master_scan_dist_list = args
    conv = None
    # for i in range(master_scan_dist_list.__len__()):
    p_fid = 'cad_indices/' + fid.split('/')[1].split('.')[0] + '.pkl'
    # if fid.split('/')[1].split('.')[0] + '.pkl' not in os.listdir("cad_indices"):
    # p = create_conv_image_indices(trimesh.load(fid), 15, 0.75, p_fid)
    # else:
    with open(p_fid, 'rb') as f:
        p = pickle.load(f)


    if conv is None:
        conv = create_conv_image_from_indices(p, master_scan_dist_list, 15, False)
    else:
        conv = torch.cat((conv,
                          create_conv_image_from_indices(p, master_scan_dist_list, 15, False)), axis=0)
    return conv

def combine_results(results):
    for i in range(results.__len__()):
        if i == 0:
            master_conv = results[i].reshape(-1, 15, 15)
        else:
            master_conv = torch.cat((master_conv, results[i].reshape(-1, 15, 15)), axis=0)
    return master_conv


def create_conv_data(master_scan_dist_list, master_ref_mesh_list):
    master_conv = []
    for i in tqdm(range(len(master_ref_mesh_list))):
        # master_scan_dist_list[i].flags.writeable = False
        hsh = str(master_scan_dist_list[i].sum().round(3))
        if os.path.isfile(f"cad_conv/{hsh}.pkl"):
            conv = torch.load(f"cad_conv/{hsh}.pkl")
        else:
            p_fid = 'cad_indices/' + master_ref_mesh_list[i].split('/')[1].split('.')[0] + '.pkl'
            with open(p_fid, 'rb') as f:
                p = pickle.load(f)
            conv = create_conv_image_from_indices(p, master_scan_dist_list[i], 15, False)
            torch.save(conv, f"cad_conv/{hsh}.pkl")
        master_conv.append(conv)

    return master_conv

def create_conv_image_indices(ref_mesh_heavy, ref_mesh_light, shape=15, step=0.75, f_id = 'p.pkl'):
    # Calculate sharp edges and other values

    # === multiprocessed ===
    # print('Starting pool of processes')
    # # Create a pool of processes
    # with Pool() as pool:
    #     all_scan_case_dist = pool.map(MP_compute_indices,
    #                                    [(i, vert, ref_mesh, shape,
    #                                      step, SE, k_tree) for i, vert in enumerate(ref_mesh.vertices)])

    # # Combine the results from each process
    # all_scan_case_dist = [item for sublist in results for item in sublist]

    # === Single process ===
    # all_scan_case_dist = compute_indices(ref_mesh, shape, step, SE, k_tree)

    # Save the results to a file and return them

    all_scan_case_dist = alternate_compute_indices(ref_mesh_light, ref_mesh_heavy, 15, 0.1)

    if f_id is not None:
        with open(f_id, 'wb') as f:
            pickle.dump(all_scan_case_dist, f)
            print('saved')
    return all_scan_case_dist

def alternate_compute_indices(ref_mesh_light, ref_mesh_heavy, shape, step):
    k_tree = ref_mesh_heavy.kdtree
    SE, _, _, _, _, _ = igl.sharp_edges(ref_mesh_heavy.vertices, ref_mesh_heavy.faces, 1.4)
    all_scan_case_dist = []
    for i in tqdm(range(ref_mesh_light.vertices.shape[0])):
        vi = ref_mesh_light.vertices[i]
        ni = ref_mesh_light.vertex_normals[i]

        if np.abs(ni).argmax() == 2:
            ori = vi - np.array((0, 0, vi[2]))
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
                v = vi - x - y
                p = k_tree.query_ball_point(v, step)
                if len(p) != 0:
                    p = list(set(p) - set(list(np.unique(SE))))
                    p = np.asarray(p)[np.nonzero(ref_mesh_heavy.vertex_normals[p, :].dot(ni) > 0.67)]
                if len(p) == 0:
                    scan_case_dist.append(-1)
                else:
                    scan_case_dist.append(p)

        all_scan_case_dist.append(scan_case_dist)

    # Return list of computed indices for all vertices in the chunk
    return all_scan_case_dist


def MP_compute_indices(args):
    i, vert, ref_mesh, shape, step, SE, k_tree = args
    ni = ref_mesh.vertex_normals[i, :]

    if np.abs(ni).argmax() == 2:
        ori = vert - np.array((0, 0, vert[2]))
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
            v = vert - x - y
            p = k_tree.query_ball_point(v, step)
            if len(p) != 0:
                p = list(set(p) - set(list(np.unique(SE))))
                p = np.asarray(p)[np.nonzero(ref_mesh.vertex_normals[p, :].dot(ni) > 0.67)]
            if len(p) == 0:
                scan_case_dist.append(-1)
            else:
                scan_case_dist.append(p)

    return [scan_case_dist, i]


def compute_indices(ref_mesh, shape, step, SE, k_tree):
    all_scan_case_dist = []

    # Compute indices for vertices in chunk
    for i, vert in tqdm(enumerate(ref_mesh.vertices), total=len(ref_mesh.vertices)):
        ni = ref_mesh.vertex_normals[i, :]

        # Creation of the y axis of the conv cases, either toward the center of the mesh or the top
        if np.abs(ni).argmax() == 2:
            ori = vert - np.array((0, 0, vert[ 2]))
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
                v = vert - x - y
                p = k_tree.query_ball_point(v, step)
                if len(p) != 0:
                    p = list(set(p) - set(list(np.unique(SE))))
                    p = np.asarray(p)[
                        np.nonzero(ref_mesh.vertex_normals[p, :].dot(ni) > 0.67)]
                if len(p) == 0:
                    scan_case_dist.append(-1)
                else:
                    scan_case_dist.append(p)
        all_scan_case_dist.append(scan_case_dist)

    # Return list of computed indices for all vertices in the chunk
    return all_scan_case_dist

def create_conv_image_from_indices(indices, scan_dist,shape=15, show_p_bar=True):
    l_scan_case_dist = []
    scan_case_dist = np.ones((shape, shape))
    if show_p_bar:
        for u, ind in enumerate(tqdm(indices)):
            for i, p in enumerate(ind):
                if np.any(p == -1):
                    scan_case_dist[i % shape, i // shape] = 0
                else:
                    scan_case_dist[i % shape, i // shape] = scan_dist[p].mean()
            l_scan_case_dist.append(scan_case_dist.copy())
    else:
        for u, ind in enumerate(indices):
            for i, p in enumerate(ind):
                if np.any(p == -1):
                    scan_case_dist[i % shape, i // shape] = 0
                else:
                    scan_case_dist[i % shape, i // shape] = scan_dist[p].mean()
            l_scan_case_dist.append(scan_case_dist.copy())
    return torch.tensor(np.array(l_scan_case_dist))

def single_conv_image(scan_dist, ref_mesh):
    P = create_conv_image_indices(ref_mesh, f_id=None, step=0.75)
    conv = create_conv_image_from_indices(P, scan_dist)
    return conv


if __name__ == '__main__':
    fid = 'cad_model/mod_nist_light_in_process.stl'
    p_fid = 'cad_indices/' + fid.split('/')[1].split('.')[0] + '.pkl'
    p = create_conv_image_indices(trimesh.load(fid), 15, 0.75, p_fid)
    fid_light = 'cad_model/mod_nist_light.stl'
    fid_heavy = 'cad_model/mod_nist.stl'
    p_fid = 'cad_indices/' + fid_heavy.split('/')[1].split('.')[0] + '.pkl'
    p = create_conv_image_indices(trimesh.load(fid_heavy), trimesh.load(fid_light), 15, 0.75, p_fid)
    fid = 'cad_model/big_test_part_1_light.stl'
    p_fid = 'cad_indices/' + fid.split('/')[1].split('.')[0] + '.pkl'
    p = create_conv_image_indices(trimesh.load(fid), 15, 0.75, p_fid)
    fid = 'cad_model/test_part_2_light.stl'
    p_fid = 'cad_indices/' + fid.split('/')[1].split('.')[0] + '.pkl'
    p = create_conv_image_indices(trimesh.load(fid), 15, 0.75, p_fid)
    fid = 'cad_model/test_part_3_light.stl'
    p_fid = 'cad_indices/' + fid.split('/')[1].split('.')[0] + '.pkl'
    p = create_conv_image_indices(trimesh.load(fid), 15, 0.75, p_fid)
    # fid = 'cad_model/test_part_4_light.stl'
    # p_fid = 'cad_indices/' + fid.split('/')[1].split('.')[0] + '.pkl'
    # p = create_conv_image_indices(trimesh.load(fid), 15, 0.50, p_fid)
    fid = 'cad_model/test_part_5_light.stl'
    p_fid = 'cad_indices/' + fid.split('/')[1].split('.')[0] + '.pkl'
    p = create_conv_image_indices(trimesh.load(fid), 15, 0.75, p_fid)
    fid = 'cad_model/test_part_6_light.stl'
    p_fid = 'cad_indices/' + fid.split('/')[1].split('.')[0] + '.pkl'
    p = create_conv_image_indices(trimesh.load(fid), 15, 0.75, p_fid)

    print('indices done')
    # raise Exception('done')

    # === Train dataprep ===
    print("started train dataprep\n")
    master_scan_dist_heavy_list, master_ave_dist_list, master_ref_mesh_list, master_scan_dist_list = create_scan_dist(mode='train')

    with open('temp/master_scan_dist_heavy_list.pkl', 'wb') as f:
        pickle.dump(master_scan_dist_heavy_list, f)
    with open('temp/master_ave_dist_list.pkl', 'wb') as f:
        pickle.dump(master_ave_dist_list, f)
    with open('temp/master_ref_mesh_list.pkl', 'wb') as f:
        pickle.dump(master_ref_mesh_list, f)
    with open('temp/master_scan_dist_list.pkl', 'wb') as f:
        pickle.dump(master_scan_dist_list, f)

    with open('temp/master_scan_dist_heavy_list.pkl', 'rb') as f:
        master_scan_dist_heavy_list = pickle.load(f)
    with open('temp/master_ave_dist_list.pkl', 'rb') as f:
        master_ave_dist_list = pickle.load(f)
    with open('temp/master_ref_mesh_list.pkl', 'rb') as f:
        master_ref_mesh_list = pickle.load(f)
    with open('temp/master_scan_dist_list.pkl', 'rb') as f:
        master_scan_dist_list = pickle.load(f)

    master_conv = create_conv_data(master_scan_dist_heavy_list, master_ref_mesh_list)
    master_conv = torch.cat(master_conv)
    torch.save(master_conv, "data/master_conv_with_mean.trc")
    print('train dataprep done')


    # === Test dataprep ===
    print("Started test dataprep\n")
    master_scan_dist_list, master_ave_dist_list, master_ref_mesh_list, master_scan_dist_light_list = create_scan_dist(mode='test')

    with open('temp/test_master_scan_dist_list_heavy.pkl', 'wb') as f:
        pickle.dump(master_scan_dist_list, f)
    with open('temp/test_master_ave_dist_list.pkl', 'wb') as f:
        pickle.dump(master_ave_dist_list, f)
    with open('temp/test_master_ref_mesh_list.pkl', 'wb') as f:
        pickle.dump(master_ref_mesh_list, f)
    with open('temp/test_master_scan_dist_list.pkl', 'wb') as f:
        pickle.dump(master_scan_dist_light_list, f)


    with open('temp/test_master_scan_dist_list_heavy.pkl', 'rb') as f:
        master_scan_dist_list = pickle.load(f)
    with open('temp/test_master_ave_dist_list.pkl', 'rb') as f:
        master_ave_dist_list = pickle.load(f)
    with open('temp/test_master_ref_mesh_list.pkl', 'rb') as f:
        master_ref_mesh_list = pickle.load(f)
    with open('temp/test_master_scan_dist_list.pkl', 'rb') as f:
        master_scan_dist_light_list = pickle.load(f)

    master_conv = create_conv_data(master_scan_dist_list, master_ref_mesh_list)
    master_conv = torch.cat(master_conv)
    torch.save(master_conv, "data/test_master_conv_with_mean.trc")
    print('test dataprep done')
