import os

import torch
import trimesh
from torch.nn import *
from torch.optim import *
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
import copy
import pickle


class homemade_cnn(Module):
    def __init__(self, step=1, n_case=15, batch_size=2, device=torch.device("cpu")):
        super(homemade_cnn, self).__init__()
        self.step = step
        self.n_case = n_case
        self.batch_size = batch_size
        self.device = device

        self.norm = BatchNorm2d(1)
        self.c1 = Conv2d(1, 64, (5, 5))
        self.r1 = ReLU()
        self.c2 = Conv2d(64, 128, (5, 5))
        self.r2 = ReLU()
        self.c3 = Conv2d(128, 256, (3, 3))
        self.r3 = ReLU()
        self.c4 = Conv2d(256, 512, (3, 3))
        self.r4 = ReLU()
        self.c5 = Conv2d(512, 1024, (3, 3))
        self.r5 = ReLU()
        self.c6 = Conv2d(1024, 512, (1, 1))
        self.r6 = ReLU()
        self.c7 = Conv2d(512, 256, (1, 1))
        self.r7 = ReLU()
        self.c8 = Conv2d(256, 128, (1, 1))
        self.r8 = ReLU()
        self.c9 = Conv2d(128, 64, (1, 1))
        self.r9 = ReLU()
        self.Lin1 = Linear(64, 8)
        self.Lin2 = Linear(9,1)

    def forward(self, input, input2, in_training=False):

        # y = self.norm(input.reshape(-1, 1, 15, 15))
        y = input.reshape(-1, 1, 15, 15)
        y = self.r1(self.c1(y))
        y = self.r2(self.c2(y))
        y = self.r3(self.c3(y))
        y = self.r4(self.c4(y))
        y = self.r5(self.c5(y))
        y = self.r6(self.c6(y))
        y = self.r7(self.c7(y))
        y = self.r8(self.c8(y))
        y = self.r9(self.c9(y)).reshape((-1, 64))
        y = self.Lin1(y).reshape(-1, 8)
        y = torch.cat((y, input2[:, None]), 1).reshape(-1, 9)
        y = self.Lin2(y).reshape(-1)

        return y

    def train_loop(self, Data, loss_fn, optimizer, epoch):
        Loss = 0
        test = 0
        for data in Data:
            optimizer.zero_grad()
            x_data, x2_data, y_data = data
            x_data, x2_data, y_data = x_data.to(self.device), x2_data.to(self.device), y_data.to(self.device)
            pred = self.forward(x_data, x2_data)
            loss = loss_fn(pred, y_data)
            # print(loss.item())
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]

            # Backpropagation
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.item())
            wandb.log({"Train loss": loss.item(), "epoch": epoch})

        Loss = Loss/test

        return Loss

    def test_loop(self, Data, loss_fn, epoch):
        Loss = 0
        test = 0
        for data in Data:
            x_data, x2_data, y_data = data
            x_data, x2_data, y_data = x_data.to(self.device), x2_data.to(self.device), y_data.to(self.device)
            pred = self.forward(x_data, x2_data)
            loss = loss_fn(pred, y_data)
            # print(loss.item())
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]
            wandb.log({"Test loss": loss.item(), "epoch": epoch})

        Loss = Loss / test

        return Loss


class dataset(torch.utils.data.IterableDataset):
    def __init__(self, X, X2, Y):
        self.X = X
        self.Y = Y
        self.X2 = X2

    def __iter__(self):
        return zip(self.X, self.X2, self.Y)

    def __len__(self):
        return len(self.X)

def train_generalized_CNN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 10000
    lr = 1e-3
    max_epoch = 500
    wandb.init(project='generalized CNN', mode='online')
    wandb.config = {"learning_rate": lr, "epochs": max_epoch, "batch_size": batch_size}
    if wandb.run.name is None:
        wandb.run.name = 'offline_test'

    l_fn = MSELoss(reduction='mean')

    l_scan_case_dist = torch.load("data/master_conv.trc").type(torch.float)
    ind = torch.where(l_scan_case_dist.sum(axis=1).sum(axis=1) != 0)[0]
    l_scan_case_dist = l_scan_case_dist[ind, :, :]

    with open('temp/master_ave_dist_list.pkl', 'rb') as f:
        ave_dist = pickle.load(f)
    for i in range(ave_dist.__len__()):
        if i == 0:
            temp = ave_dist[i].reshape(-1)
        else:
            temp = np.concatenate((temp, ave_dist[i].reshape(-1)), axis=0)
    ave_dist = temp
    ave_dist = torch.from_numpy(np.array(ave_dist)).type(torch.float)[ind]

    with open('temp/master_scan_dist_list.pkl', 'rb') as f:
        center_dist = pickle.load(f)
    for i in range(center_dist.__len__()):
        if i == 0:
            temp = center_dist[i].reshape(-1)
        else:
            temp = np.concatenate((temp, center_dist[i].reshape(-1)), axis=0)
    center_dist = temp
    center_dist = torch.from_numpy(np.array(center_dist)).type(torch.float)[ind]

    ind = torch.where(center_dist!=0)[0]
    l_scan_case_dist = l_scan_case_dist[ind, :, :]
    center_dist = center_dist[ind]
    ave_dist = ave_dist[ind]

    seed = torch.manual_seed(0)
    train_gt, test_gt = torch.utils.data.random_split(center_dist, [int(np.round(center_dist.shape[0]*0.75)),
                                                                    int(np.round(center_dist.shape[0]*0.25))], seed)
    x_train, x_test = l_scan_case_dist[train_gt.indices], l_scan_case_dist[test_gt.indices]
    x2_train, x2_test = center_dist[train_gt.indices], center_dist[test_gt.indices]
    y_train, y_test = ave_dist[train_gt.indices], ave_dist[test_gt.indices]

    # train_gt, test_gt = range(int(np.round(center_dist.shape[0] * 0.75))), range(
    #     int(np.round(center_dist.shape[0] * 0.25)))
    # x_train, x_test = l_scan_case_dist[train_gt], l_scan_case_dist[test_gt]
    # x2_train, x2_test = center_dist[train_gt], center_dist[test_gt]
    # y_train, y_test = ave_dist[train_gt], ave_dist[test_gt]
    #
    train_dataset = dataset(x_train, x2_train, y_train)
    test_dataset = dataset(x_test, x2_test, y_test)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    # hmc = homemade_cnn(batch_size=batch_size, device=device).to(device)
    hmc = torch.load('NN_model/worthy-totem-6model.trc')
    opt = AdamW(hmc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, verbose=True)
    wandb.watch(hmc, log_freq=10)

    test_loss = np.inf
    best_test_loss = np.inf

    pbar = tqdm(range(max_epoch), desc="test loss = " + str(test_loss) + " Train_loss = " + str(test_loss))
    for epoch in pbar:
        hmc.train()
        train_loss = hmc.train_loop(train_data, l_fn, opt, epoch)
        wandb.log({"Mean train loss": train_loss, "epoch": epoch})
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))

        hmc.eval()
        test_loss = hmc.test_loop(test_data, l_fn, epoch)
        wandb.log({"Mean test loss": test_loss, "epoch": epoch})
        scheduler.step(test_loss)
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))
        wandb.log({'epoch': epoch, 'Learning rate': opt.param_groups[0]['lr']})

        if test_loss <= best_test_loss:
            bestmodel = copy.deepcopy(hmc)
            best_test_loss = test_loss
            bestmodel_epoch = epoch

        if epoch % 10 ==1:
            torch.save(bestmodel, "NN_model/" + wandb.run.name + 'model.trc')
            print('saved best model with loss ' + str(best_test_loss) + ' at epoch +' + str(bestmodel_epoch))


    torch.save(bestmodel, "NN_model/" + wandb.run.name + 'model.trc')
    print('saved best model with loss ' + str(best_test_loss) + ' at epoch +' + str(bestmodel_epoch))

    val_train_loss = bestmodel.test_loop(train_data, l_fn, epoch)
    val_test_loss = bestmodel.test_loop(test_data, l_fn, epoch)
    print('best model training loss: ' + str(val_train_loss))
    print('best model test loss: ' + str(val_test_loss))

def nn_compensate(nn_model_fid, dist, ref_mesh_fid):
    import dataprep
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    p_fid = ref_mesh_fid.split('/')[1].split('.')[0] + '.pkl'
    if p_fid in os.listdir("cad_indices"):
        with open('cad_indices/' + p_fid, 'rb') as f:
            p = pickle.load(f)
        conv = dataprep.create_conv_image_from_indices(p, dist, show_p_bar=False).type(torch.float)
    else:
        ref_mesh = trimesh.load_mesh(ref_mesh_fid)
        conv = dataprep.single_conv_image(dist, ref_mesh).type(torch.float)
        
    model = torch.load(nn_model_fid).type(torch.float).to(device)
    
    
    ddataset = dataset(conv, torch.tensor(dist), torch.tensor(dist))
    Data = DataLoader(ddataset, batch_size=10000)
    pred = torch.zeros_like(torch.tensor(dist))
    for i, data in enumerate(Data):
        x_data, x2_data, y_data = data
        x_data, x2_data, y_data = x_data.to(device).type(torch.float), x2_data.to(device).type(torch.float), y_data.to(device).type(torch.float)
        pred[i*10000:(i+1)*10000] = model.forward(x_data, x2_data) 

    return pred