import easygui
from torch.nn import *
from torch.optim import *
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
import copy
import pickle
import model
import os
import trimesh
from utils import measure_distance
import csv
import torch
import logging
logging.getLogger("trimesh").setLevel(logging.ERROR)


def analyse(nn_fid=None):
    if nn_fid is None:
        nn_fid = easygui.fileopenbox('Select a nn to test', 'Select a nn to test')
    
    csv_result_fid = 'result/' + nn_fid[:-4] + '.csv'
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
        for subdir_id in tqdm(os.listdir(dir_id)):
            ref_mesh_fid = "cad_model/" + subdir_id.split('.')[0] + ".stl"
            ref_mesh = trimesh.load(ref_mesh_fid)
            subdir_id = dir_id + subdir_id + "/"

            l_dist, l_nn_dist, l_scan_mesh = [], [], []
            for f_id in os.listdir(subdir_id):
                f_id = subdir_id + f_id
                l_scan_mesh.append(f_id)
                scan_mesh = trimesh.load(f_id)
                _, dist = measure_distance(scan_mesh, ref_mesh)
                l_dist.append(dist)

                nn_dist = model.nn_compensate(nn_fid, dist, ref_mesh_fid)
                l_nn_dist.append(nn_dist)

            ave_dist = np.mean(np.array(l_dist), axis=0)

            for i in range(len(l_dist)):
                csvwriter.writerow([
                    ref_mesh_fid, l_scan_mesh[i], np.mean(np.abs(l_dist[i])), np.mean(np.abs(ave_dist)),
                    np.mean(np.abs(l_nn_dist[i].numpy())), np.mean(np.abs(l_dist[i]-ave_dist)),
                    np.mean(np.abs(l_nn_dist[i].numpy() - ave_dist)), np.std(l_dist[i]), np.std(ave_dist), np.std(l_nn_dist[i].numpy()),
                    np.std(l_dist[i]-ave_dist), np.std(l_nn_dist[i].numpy()-ave_dist)
                ])


if __name__ == '__main__':
    analyse('NN_model/restful-blaze-14model.trc')
