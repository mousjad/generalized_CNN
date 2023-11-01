import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np
from tqdm import tqdm
from utils import git_push
import wandb
from pointconv_utils import PointConvDensitySetAbstraction, prepare_data

class Pointconv(torch.nn.Module):
    def __init__(self, device):
        super(Pointconv, self).__init__()
        feature_dim = 3
        self.device = device
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + 3, mlp=[64, 64, 128],
                                                  bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                                  bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024],
                                                  bandwidth = 0.4, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

    def train_loop(self, data, optimizer):
        for data in data:
            self.train()
            train_dist, train_xyz, train_mean = data
            train_dist, train_xyz, train_mean = train_dist.to(self.device), train_xyz.to(self.device), train_mean.to(self.device)
            train_dist = train_dist.reshape((-1, 1, 1))
            train_xyz = train_xyz.reshape((-1, 3, 1))
            pred = self.forward(train_xyz, train_dist)
            loss = torch.nn.functional.mse_loss(pred.squeeze(), train_mean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    def test_loop(self, data):
        for data in data:
            self.eval()
            test_dist, test_xyz, test_mean = data
            test_dist, test_xyz, test_mean = test_dist.to(self.device), test_xyz.to(self.device), test_mean.to(self.device)
            pred = self.forward(test_xyz, test_dist)
            loss = torch.nn.functional.mse_loss(pred.squeeze(), test_mean)

        return loss.item()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Pointconv(device).float().to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 500
    max_epoch = 100

    # git_push(r"C:\Generalized_CNN\.git", f"{wandb.run.name}_automated_commit")

    # === train data import ===
    train_dist, train_xyz, train_mean = prepare_data("train")

    # === Test data import ===
    test_dist, test_xyz, test_mean = prepare_data("test")

    model(train_xyz[0, 0, :512].reshape((1, 3, 512)).to(device), train_xyz[0, 0, :512].reshape((1, 3, 512)).to(device))

    train_dataset = dataset(train_dist, train_xyz, train_mean)
    test_dataset = dataset(test_dist, test_xyz, test_mean)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    # Training loop
    wandb.init(project='Pointconv', mode='offline')
    wandb.config = {"learning_rate": lr, "epochs": max_epoch, "batch_size": batch_size}
    if wandb.run.name is None:
        wandb.run.name = 'offline_test'
    test_loss = torch.inf
    train_loss = torch.inf
    pbar = tqdm(range(max_epoch), desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))
    for epoch in pbar:
        # === train ===
        train_loss = model.train_loop(train_data, optimizer)
        wandb.log({"Mean train loss": train_loss, "epoch": epoch})
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))

        # === test ===
        test_loss = model.test_loop(test_data)
        wandb.log({"Mean train loss": train_loss, "epoch": epoch})
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))


class dataset(torch.utils.data.IterableDataset):
    def __init__(self, X, X2, Y):
        self.X = X
        self.X2 = X2
        self.Y = Y

    def __iter__(self):
        return zip(self.X, self.X2, self.Y)

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    main()
