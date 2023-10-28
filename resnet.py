import torch
from torch.nn import *
from torch.optim import *
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from tempfile import TemporaryDirectory
import wandb, pickle
from tqdm import tqdm
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize([0.485], [0.229])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.Normalize([0.485], [0.229])
    ]),
}

class ResNet(Module):
    def __init__(self, device):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18()
        self.device = device

        # Remove last layer (classifier)
        nb_ftrs = self.resnet.fc.in_features
        self.resnet = Sequential(*list(self.resnet.children())[:-1])

        #Define new layers
        self.conv0 = Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.fc1 = Linear(nb_ftrs, 1)

    def forward(self, input1):
        x = self.conv0(input1)
        x = self.resnet(x).view((-1, 512))
        x = torch.flatten(self.fc1(x))

        return x

    def train_loop(self, Data, loss_fn, optimizer, epoch):
        Loss = 0
        test = 0
        for data in Data:
            optimizer.zero_grad()
            x_data, y_data = data
            x_data, y_data = x_data.to(self.device), y_data.to(self.device)
            pred = self.forward(data_transforms['train'](x_data))
            loss = loss_fn(pred, y_data)
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]

            # Backpropagation
            loss.backward()
            optimizer.step()
            wandb.log({"Train loss": loss.item(), "epoch": epoch})

        Loss = Loss/test

        return Loss

    def test_loop(self, Data, loss_fn, epoch):
        Loss = 0
        test = 0
        for data in Data:
            x_data, y_data = data
            x_data, y_data = x_data.to(self.device), y_data.to(self.device)
            pred = self.forward(data_transforms['val'](x_data))
            loss = loss_fn(pred, y_data)
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]
            wandb.log({"Test loss": loss.item(), "epoch": epoch})

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




def train_resnet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 250
    lr = 1e-3
    max_epoch = 100
    wandb.init(project='generalized CNN resnet', mode='online')
    wandb.config = {"learning_rate": lr, "epochs": max_epoch, "batch_size": batch_size}
    if wandb.run.name is None:
        wandb.run.name = 'offline_test'

    l_fn = MSELoss(reduction='mean')

    l_scan_case_dist = torch.load("data/master_conv_with_mean.trc").type(torch.float)

    with open('temp/master_ave_dist_list.pkl', 'rb') as f:
       ave_dist = pickle.load(f)
    for i in range(ave_dist.__len__()):
       if i == 0:
           temp = ave_dist[i].reshape(-1)
       else:
           temp = np.concatenate((temp, ave_dist[i].reshape(-1)), axis=0)
    ave_dist = temp
    ave_dist = torch.from_numpy(np.array(ave_dist)).type(torch.float)

    with open('temp/master_scan_dist_list.pkl', 'rb') as f:
       center_dist = pickle.load(f)
    for i in range(center_dist.__len__()):
       if i == 0:
           temp = center_dist[i].reshape(-1)
       else:
           temp = np.concatenate((temp, center_dist[i].reshape(-1)), axis=0)
    center_dist = temp
    center_dist = torch.from_numpy(np.array(center_dist)).type(torch.float)

    ind = torch.where(center_dist != 0)[0]
    x_train = l_scan_case_dist[ind, :, :]
    x2_train = center_dist[ind]
    y_train = ave_dist[ind]

    x_train = x_train.reshape((-1, 1, 15, 15))
    # x_train_mask = torch.zeros_like(x_train)
    # x_train_mask[torch.where(x_train != 0)] = 1
    # x_train = torch.cat((x_train.reshape((-1, 1, 15, 15)), x_train_mask.reshape((-1, 1, 15, 15))), dim=1)

    sum = x_train.sum(axis=(2, 3))
    train_filt_max = np.percentile(sum, 99)
    train_filt_min = np.percentile(sum, 1)
    filt1 = (sum <= train_filt_max)
    filt2 = (sum >= train_filt_min)
    filt = (filt1) & (filt2)
    x_train = x_train[torch.nonzero(filt[:, 0])[:, 0]]
    x2_train = x2_train[torch.nonzero(filt[:, 0])[:, 0]]
    y_train = y_train[torch.nonzero(filt[:, 0])[:, 0]]

    diff = x2_train - y_train
    train_filt_max = np.percentile(diff, 99)
    train_filt_min = np.percentile(diff, 1)
    filt1 = (diff <= train_filt_max)
    filt2 = (diff >= train_filt_min)
    filt = (filt1) & (filt2)
    x_train = x_train[torch.nonzero(filt)[:, 0]]
    x2_train = x2_train[torch.nonzero(filt)[:, 0]]
    y_train = y_train[torch.nonzero(filt)[:, 0]]

    filt = torch.where(x_train != 0)
    x_train[filt] = x_train[filt] + 0.5

    torch.save(x_train, "data/x_train.trc")
    torch.save(x2_train, "data/x2_train.trc")
    torch.save(y_train, "data/y_train.trc")

    x_train = torch.load("data/x_train.trc")
    y_train = torch.load("data/y_train.trc")

    idx = torch.randperm(x_train.size(0))
    x_train = x_train[idx]
    y_train = y_train[idx]

    # === Test data import ===
    l_scan_case_dist = torch.load("data/test_master_conv_with_mean.trc").type(torch.float)


    with open('temp/test_master_ave_dist_list.pkl', 'rb') as f:
       ave_dist = pickle.load(f)
    for i in range(ave_dist.__len__()):
       if i == 0:
           temp = ave_dist[i].reshape(-1)
       else:
           temp = np.concatenate((temp, ave_dist[i].reshape(-1)), axis=0)
    ave_dist = temp
    ave_dist = torch.from_numpy(np.array(ave_dist)).type(torch.float)

    with open('temp/test_master_scan_dist_list.pkl', 'rb') as f:
       center_dist = pickle.load(f)
    for i in range(center_dist.__len__()):
       if i == 0:
           temp = center_dist[i].reshape(-1)
       else:
           temp = np.concatenate((temp, center_dist[i].reshape(-1)), axis=0)
    center_dist = temp
    center_dist = torch.from_numpy(np.array(center_dist)).type(torch.float)

    ind = torch.where(center_dist != 0)[0]
    x_test = l_scan_case_dist[ind, :, :]
    x2_test = center_dist[ind]
    y_test = ave_dist[ind]
    #
    x_test = x_test.reshape((-1, 1, 15, 15))
    # x_test_mask = torch.zeros_like(x_test)
    # x_test_mask[torch.where(x_test != 0)] = 1
    # x_test = torch.cat((x_test.reshape((-1, 1, 15, 15)), x_test_mask.reshape((-1, 1, 15, 15))), dim=1)
    #

    # Filtering
    sum = x_test.sum(axis=(2, 3))
    test_filt_max = np.percentile(sum, 99)
    test_filt_min = np.percentile(sum, 1)
    filt1 = (sum <= test_filt_max)
    filt2 = (sum >= test_filt_min)
    filt = (filt1) & (filt2)
    x_test = x_test[torch.nonzero(filt[:, 0])[:, 0]]
    x2_test = x2_test[torch.nonzero(filt[:, 0])[:, 0]]
    y_test = y_test[torch.nonzero(filt[:, 0])[:, 0]]

    diff = x2_test - y_test
    test_filt_max = np.percentile(diff, 99)
    test_filt_min = np.percentile(diff, 1)
    filt1 = (diff <= test_filt_max)
    filt2 = (diff >= test_filt_min)
    filt = (filt1) & (filt2)
    x_test = x_test[torch.nonzero(filt)[:, 0]]
    x2_test = x2_test[torch.nonzero(filt)[:, 0]]
    y_test = y_test[torch.nonzero(filt)[:, 0]]

    # filt = torch.where(x_test != 0)
    # x_test[filt] = x_test[filt] + 0.5

    torch.save(x_test, "data/x_test.trc")
    torch.save(x2_test, "data/x2_test.trc")
    torch.save(y_test, "data/y_test.trc")

    x_test = torch.load("data/x_test.trc")
    y_test = torch.load("data/y_test.trc")
    idx = torch.randperm(x_test.size(0))
    x_test = x_test[idx]
    y_test = y_test[idx]

    # x_test = torch.load("data/subset_x_test.trc")
    # x2_test = torch.load("data/subset_x2_test.trc")
    # y_test = torch.load("data/subset_y_test.trc")
    # x_train = torch.load("data/subset_x_train.trc")
    # x2_train = torch.load("data/subset_x2_train.trc")
    # y_train = torch.load("data/subset_y_train.trc")

    train_dataset = dataset(x_train, y_train)
    test_dataset = dataset(x_test, y_test)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    model = ResNet(device).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, verbose=True)
    wandb.watch(model, log_freq=10)

    test_loss = np.inf
    best_test_loss = np.inf

    pbar = tqdm(range(max_epoch), desc="test loss = " + str(test_loss) + " Train_loss = " + str(test_loss))
    for epoch in pbar:
        model.train()
        train_loss = model.train_loop(train_data, l_fn, opt, epoch)
        wandb.log({"Mean train loss": train_loss, "epoch": epoch})
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))

        model.eval()
        test_loss = model.test_loop(test_data, l_fn, epoch)
        wandb.log({"Mean test loss": test_loss, "epoch": epoch})
        scheduler.step(test_loss)
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))
        wandb.log({'epoch': epoch, 'Learning rate': opt.param_groups[0]['lr']})

        if test_loss <= best_test_loss:
            bestmodel = copy.deepcopy(model)
            best_test_loss = test_loss
            bestmodel_epoch = epoch

        if epoch % 10 ==1:
            torch.save(bestmodel, "NN_model/" + wandb.run.name + 'model.trc')
            print('saved best model with loss ' + str(best_test_loss) + ' at epoch +' + str(bestmodel_epoch))
            # train_dataset.current_subset_size += int(np.round(len(x_train)*0.1))
            # test_dataset.current_subset_size += int(np.round(len(x_test)*0.1))


    torch.save(bestmodel, "NN_model/" + wandb.run.name + 'model.trc')
    print('saved best model with loss ' + str(best_test_loss) + ' at epoch +' + str(bestmodel_epoch))

    val_train_loss = bestmodel.test_loop(train_data, l_fn, epoch)
    val_test_loss = bestmodel.test_loop(test_data, l_fn, epoch)
    print('best model training loss: ' + str(val_train_loss))
    print('best model test loss: ' + str(val_test_loss))
    return "NN_model/" + wandb.run.name + 'model.trc'

if __name__ == '__main__':
    train_resnet()
