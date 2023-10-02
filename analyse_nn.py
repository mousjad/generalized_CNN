import easygui
from tqdm import tqdm
import numpy as np
import model
import os
import trimesh
from utils import measure_distance
import csv
from draw_compare import create_image
import igl
import logging
logging.getLogger("trimesh").setLevel(logging.ERROR)


def analyse(nn_fid=None):
    if nn_fid is None:
        nn_fid = easygui.fileopenbox('Select a nn to test', 'Select a nn to test')
    
    csv_result_fid = 'result/' + nn_fid[9:-4] + '.csv'
    with open(csv_result_fid, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            nn_fid
        ])
        csvwriter.writerow([
            'ref_mesh', 'scan_mesh', 'Absolute deviation mean of single scan', 'Absolute deviation mean of average scan',
            'Absolute deviation mean of CNN output', 'Absolute deviation mean of (single scan - average scan)',
            'Absolute deviation mean of (CNN output - average scan)',
            'STD of single scan', 'STD of average scan', 'STD of CNN output', 'STD of (single scan - average scan)',
            'STD of (CNN output - average scan)'
        ])



        dir_id="scan_data/"

        if not (os.path.exists('result/' + nn_fid[9:-4]) and os.path.isdir('result/' + nn_fid[9:-4])):
            os.mkdir('result/' + nn_fid[9:-4])
        for subdir_id in tqdm(os.listdir(dir_id)):
            if subdir_id[0] != '.':
                ref_mesh_fid = "cad_model/" + subdir_id.split('.')[0] + ".stl"
                ref_mesh = trimesh.load(ref_mesh_fid)
                subdir_id = dir_id + subdir_id + "/"
                SE, _, _, _, _, _ = igl.sharp_edges(ref_mesh.vertices, ref_mesh.faces, 1.4)

                l_dist, l_nn_dist, l_scan_mesh = [], [], []
                for f_id in os.listdir(subdir_id):
                    f_id = subdir_id + f_id
                    l_scan_mesh.append(f_id)
                    scan_mesh = trimesh.load(f_id)
                    _, dist = measure_distance(scan_mesh, ref_mesh)
                    l_dist.append(dist)

                    if os.path.exists('result/' + nn_fid[9:-4] + '/nn_' + subdir_id.split('/')[-2] + '_' + f_id.split('/')[-1]):
                        cp_nn_ref_mesh = trimesh.load('result/' + nn_fid[9:-4] + '/nn_' + subdir_id.split('/')[-2] + '_' + f_id.split('/')[-1])
                        _, nn_dist = measure_distance(cp_nn_ref_mesh, ref_mesh)
                    else:
                        nn_dist = model.nn_compensate(nn_fid, dist, ref_mesh_fid)
                        cp_nn_ref_mesh = ref_mesh.copy()
                        cp_nn_ref_mesh.vertices -= nn_dist.numpy().reshape((-1, 1))*cp_nn_ref_mesh.vertex_normals
                        cp_nn_ref_mesh.export('result/' + nn_fid[9:-4] + '/nn_' + subdir_id.split('/')[-2] + '_' + f_id.split('/')[-1])

                    l_nn_dist.append(np.array(nn_dist))

                ave_dist = np.mean(np.array(l_dist), axis=0)
                cp_ave_ref_mesh = ref_mesh.copy()
                cp_ave_ref_mesh.vertices -= np.array(nn_dist).reshape((-1, 1)) * cp_ave_ref_mesh.vertex_normals
                cp_ave_ref_mesh.export('result/' + nn_fid[9:-4] + '/ave_' + subdir_id.split('/')[-2] + '.stl')

                for i in range(len(l_dist)):
                    l_nn_dist[i][SE] = np.array(l_dist[i][SE])
                    l_nn_dist[i] = np.array(l_nn_dist[i])
                    l_nn_dist[i][np.where(l_dist[i] == 0)[0]] = l_dist[i][np.where(l_dist[i] == 0)[0]]
                    l_nn_dist[i][np.where(np.abs(ref_mesh.vertex_normals[:, 2]) == 1)[0]] = l_dist[i][np.where(np.abs(ref_mesh.vertex_normals[:, 2]) == 1)[0]]
                    csvwriter.writerow([
                        ref_mesh_fid, l_scan_mesh[i], np.mean(np.abs(l_dist[i])), np.mean(np.abs(ave_dist)),
                        np.mean(np.abs(np.array(l_nn_dist[i]))), np.mean(np.abs(l_dist[i]-ave_dist)),
                        np.mean(np.abs(np.array(l_nn_dist[i]) - ave_dist)), np.std(l_dist[i]), np.std(ave_dist), np.std(np.array(l_nn_dist[i])),
                        np.std(l_dist[i]-ave_dist), np.std(np.array(l_nn_dist[i])-ave_dist)
                    ])

                    create_image(ref_mesh, 0, l_dist[i], ave_dist, np.array(l_nn_dist[i]), offscreen=True,
                                 image_fname='result/' + nn_fid[9:-4] + '/nn_' + subdir_id.split('/')[-2] + '_' + l_scan_mesh[i].split('/')[-1].split('.')[-2] + '.jpg')

def compensate_and_save(nn_fid):
    ref_mesh_fid = 'cad_model/test_part_3_light.stl'
    scan_mesh = trimesh.load(r"\\poseidon.meca.polymtl.ca\usagers\mojad\Documents\U32-03.stl")
    ref_mesh = trimesh.load(ref_mesh_fid)
    _, dist = measure_distance(scan_mesh, ref_mesh)
    cp_ref_mesh = ref_mesh.copy()
    cp_ref_mesh.vertices += np.array(dist).reshape((-1, 1)) * cp_ref_mesh.vertex_normals
    cp_ref_mesh.export('comp_original_blade.stl')

    nn_dist = model.nn_compensate(nn_fid, dist, ref_mesh_fid)
    cp_ref_mesh = ref_mesh.copy()
    cp_ref_mesh.vertices += np.array(nn_dist).reshape((-1, 1)) * cp_ref_mesh.vertex_normals
    cp_ref_mesh.export('nn_comp_original_blade.stl')


if __name__ == '__main__':
    compensate_and_save('NN_model/noble-wood-58model.trc')
    # analyse('NN_model/noble-wood-58model.trc')



