import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
import os
from utils import measure_distance
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
logging.getLogger("trimesh").setLevel(logging.ERROR)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(MLP, self).__init__()
        self.device = device

        self.norm = nn.BatchNorm1d(6)
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(64, 32)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        return x

    def train_loop(self, Data, loss_fn, optimizer):
        Loss = 0
        test = 0

        for data in Data:
            x_data, y_data = data
            x_data, y_data = x_data.to(self.device), y_data.to(self.device)

            pred = self.forward(x_data)
            loss = loss_fn(pred, y_data)
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]

            # Backpropagation
            loss.backward()
            optimizer.step()

        Loss = Loss / test

        return Loss

    def test_loop(self, Data, loss_fn):
        Loss = 0
        test = 0

        for data in Data:
            x_data, y_data = data
            x_data, y_data = x_data.to(self.device), y_data.to(self.device)

            pred = self.forward(x_data)
            loss = loss_fn(pred, y_data)
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]

        Loss = Loss / test

        return Loss

class dataset(torch.utils.data.IterableDataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __iter__(self):
        return zip(self.X, self.Y)

    def __len__(self):
        return len(self.X)

def create_scan_dist(dir_id):
    l_dist = []

    ref_mesh = trimesh.load('cad_model/' + dir_id.split('/')[1].split('.')[0] + '.stl')
    for fid in tqdm(os.listdir(dir_id)):
        fid = dir_id + '/' + fid
        scan_mesh = trimesh.load(fid)
        _, dist = measure_distance(scan_mesh, ref_mesh)
        l_dist.append(dist)
    l_dist = torch.from_numpy(np.array(l_dist).reshape(-1, 1))
    return l_dist


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_mesh = trimesh.load('cad_model/mod_nist_light.stl')

    # Assuming input data has shape (batch_size, input_dim)
    input_dim = 6  # (x, y, z, nx, ny, nz)
    output_dim = 1  # Systematic error as output

    # Create the MLP model
    mlp = MLP(input_dim, output_dim, device).to(device)
    batch_size = 1000
    lr = 1e-5
    max_epoch = 5000
    l_fn = nn.MSELoss(reduction='mean')
    opt = optim.AdamW(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, verbose=True)

    input = torch.cat((torch.from_numpy(ref_mesh.vertices), torch.from_numpy(ref_mesh.vertex_normals)), axis=1)
    input = torch.cat((input, input, input, input, input, input), axis=0).type(torch.float)
    dist = create_scan_dist('scan_data/mod_nist_light.1').type(torch.float)


    seed = torch.manual_seed(0)
    train_gt, test_gt = torch.utils.data.random_split(dist, [int(np.floor(dist.shape[0]*0.75)), int(np.ceil(dist.shape[0]*0.25))], seed)
    x_train, x_test = input[train_gt.indices], input[test_gt.indices]
    y_train, y_test = dist[train_gt.indices], dist[test_gt.indices]

    train_dataset = dataset(x_train, y_train)
    test_dataset = dataset(x_test,  y_test)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    train_loss, test_loss = np.inf, np.inf
    pbar = tqdm(range(max_epoch), desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))
    for epoch in pbar:
        train_loss = mlp.train_loop(train_data, l_fn, opt)
        test_loss = mlp.test_loop(test_data, l_fn)
        scheduler.step(test_loss)
        pbar.set_description("test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))
    return mlp


if __name__ == '__main__':
    main()