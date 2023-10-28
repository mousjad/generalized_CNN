import os
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
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
from neighboor_padding import neighboorPadding
from torchvision import datasets, models, transforms
from  utils import git_push

cudnn.benchmark = True
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])


class homemade_cnn(Module):
    def __init__(self, step=1, n_case=15, batch_size=2, device=torch.device("cpu")):
        super(homemade_cnn, self).__init__()
        self.step = step
        self.n_case = n_case
        self.batch_size = batch_size
        self.device = device
        self.dropout_rate = 0.1
        self.mask_max_pool = MaxPool2d(3, stride=1)

        self.c1 = Conv2d(1, 8, (3, 3))
        self.p1 = MaxPool2d(3, stride=1, padding=1)
        self.r1 = LeakyReLU()
        self.norm1 = BatchNorm2d(8)
        self.drop1 = Dropout(self.dropout_rate)

        self.c2 = Conv2d(8, 16, (3, 3))
        self.p2 = MaxPool2d(3, stride=1, padding=1)
        self.r2 = LeakyReLU()
        self.norm2 = BatchNorm2d(16)
        self.drop2 = Dropout(self.dropout_rate)

        self.c3 = Conv2d(16, 32, (3, 3))
        self.p3 = MaxPool2d(3, stride=1, padding=1)
        self.r3 = LeakyReLU()
        self.norm3 = BatchNorm2d(32)
        self.drop3 = Dropout(self.dropout_rate)

        self.c4 = Conv2d(32, 64, (3, 3))
        self.p4 = MaxPool2d(3, stride=1, padding=1)
        self.r4 = LeakyReLU()
        self.norm4 = BatchNorm2d(64)
        self.drop4 = Dropout(self.dropout_rate)

        self.c5 = Conv2d(64, 128, (3, 3))
        self.p5 = MaxPool2d(3, stride=1, padding=1)
        self.r5 = LeakyReLU()
        self.norm5 = BatchNorm2d(128)
        self.drop5 = Dropout(self.dropout_rate)
        #
        self.c6 = Conv2d(128, 256, (3, 3))
        self.p6 = MaxPool2d(3, stride=1, padding=1)
        self.r6 = LeakyReLU()
        self.norm6 = BatchNorm2d(256)
        self.drop6 = Dropout(self.dropout_rate)

        # self.c7 = Conv2d(64, 32, (3, 3))
        # self.p7 = MaxPool2d(3, stride=1, padding=1)
        # self.r7 = LeakyReLU()
        # self.norm7 = BatchNorm2d(32)
        # self.drop7 = Dropout(self.dropout_rate)

        # self.c8 = Conv2d(16, 8, (3, 3))
        # self.p8 = MaxPool2d(3, stride=1, padding=1)
        # self.r8 = LeakyReLU()
        # self.norm8 = BatchNorm2d(8)
        # self.drop8 = Dropout(self.dropout_rate)

        self.Lin1 = Linear(3 * 3 * 256, 1024, bias=True)
        self.lr1 = LeakyReLU()
        self.Lin2 = Linear(1024, 512, bias=True)
        self.lr2 = LeakyReLU()
        self.Lin3 = Linear(512, 128, bias=True)
        self.lr3 = LeakyReLU()
        self.Lin4 = Linear(128, 8, bias=True)
        self.input2_drop = Dropout(0.5)
        self.lin_input2 = Linear(1, 1)
        self.Lin5 = Linear(8, 1, bias=True)


    def forward(self, input, input2, in_training=False):
        # y = neighboorPadding(input[:, 0].reshape((-1, 1, 15, 15)), input[:, 1].reshape((-1, 1, 15, 15)), 3)
        y = input[:, 0].reshape((-1, 1, 15, 15))
        # y = data_transforms["train" if self.training else "val"](y)

        y = self.drop1(self.norm1(self.r1(self.p1(self.c1(y)))))
        # mask = self.mask_max_pool(input[:, 1].reshape((-1, 1, 15, 15)))
        # y = y * mask

        y = self.drop2(self.norm2(self.r2(self.p2(self.c2(y)))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask

        y = self.drop3(self.norm3(self.r3(self.p3(self.c3(y)))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask

        y = self.drop4(self.norm4(self.r4(self.p4(self.c4(y)))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask

        y = self.drop5(self.norm5(self.r5(self.p5(self.c5(y)))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask

        y = self.drop6(self.norm6(self.r6(self.p6(self.c6(y)))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask

        # y = self.drop7(self.r7(self.c7(y)))
        # mask = self.mask_max_pool(mask)
        # y = y * mask

        # y = torch.flatten(self.drop8(self.r8(self.c8(y))), start_dim=1)
        # y = self.Lin1(y)
        # y = torch.cat((y, input2[:, None]), 1).reshape(-1, 9)
        y = self.lr1(self.Lin1(torch.flatten(y, start_dim=1)))
        y = self.lr2(self.Lin2(y))
        y = self.lr3(self.Lin3(y))
        y = self.Lin4(y)
        # y2 = torch.flatten(self.input2_drop(self.lin_input2(input2.reshape((-1, 1)))))
        y = torch.flatten(self.Lin5(y))
        # y = (y + y2) / (1 + torch.where(y2 != 0, 1, 0))

        return y


    def train_loop(self, Data, loss_fn, optimizer, epoch):
        Loss = 0
        test = 0
        for data in Data:
            optimizer.zero_grad()
            x_data, x2_data, y_data = data
            x_data, x2_data, y_data = x_data.to(self.device), x2_data.to(self.device), y_data.to(self.device)
            pred = self.forward(data_transforms(x_data), x2_data)
            loss = loss_fn(pred, y_data)
            Loss += (loss.item() - loss_fn(x2_data, y_data).item()) * x_data.shape[0]
            test += x_data.shape[0]

            # Backpropagation
            loss.backward()
            optimizer.step()
            wandb.log({"Train loss": loss.item(), "epoch": epoch})

        Loss = Loss / test

        return Loss


    def test_loop(self, Data, loss_fn, epoch):
        Loss = 0
        test = 0
        for data in Data:
            x_data, x2_data, y_data = data
            x_data, x2_data, y_data = x_data.to(self.device), x2_data.to(self.device), y_data.to(self.device)
            pred = self.forward(x_data, x2_data)
            loss = loss_fn(pred, y_data)
            Loss += (loss.item() - loss_fn(x2_data, y_data).item()) * x_data.shape[0]
            test += x_data.shape[0]
            wandb.log({"Test loss": loss.item(), "epoch": epoch})

        Loss = Loss / test

        return Loss


class dataset(torch.utils.data.IterableDataset):
    def __init__(self, X, X2, Y):
        self.X = X
        self.X2 = X2
        self.Y = Y

    def __iter__(self):
        return zip(self.X, self.X2, self.Y)

    def __len__(self):
        return len(self.X)


def train_generalized_CNN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 10000
    lr = 5e-3
    max_epoch = 1500
    wandb.init(project='generalized CNN', mode='online')
    wandb.config = {"learning_rate": lr, "epochs": max_epoch, "batch_size": batch_size}
    if wandb.run.name is None:
        wandb.run.name = 'offline_test'

    git_push(r"C:\Generalized_CNN\.git", f"{wandb.run.name}_automated_commit")
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
    x_train = l_scan_case_dist[ind]
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

    filt = torch.where(x_train.sum(axis=(2, 3)) != 0)[0]
    x_train = x_train[filt]
    x2_train = x2_train[filt]
    y_train = y_train[filt]

    torch.save(x_train, "data/x_train.trc")
    torch.save(x2_train, "data/x2_train.trc")
    torch.save(y_train, "data/y_train.trc")

    x_train = torch.load("data/x_train.trc")
    x2_train = torch.load("data/x2_train.trc")
    y_train = torch.load("data/y_train.trc")

    idx = torch.randperm(x_train.size(0))
    x_train = x_train[idx]
    x2_train = x2_train[idx]
    y_train = y_train[idx]

    # filt = torch.all(x_train.reshape((-1, 1, 15*15)) != 0, dim=2)[:, 0]
    # x_train = x_train[filt]
    # x2_train = x2_train[filt]
    # y_train = y_train[filt]

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

    filt = torch.where(x_test != 0)
    x_test[filt] = x_test[filt] + 0.5

    torch.save(x_test, "data/x_test.trc")
    torch.save(x2_test, "data/x2_test.trc")
    torch.save(y_test, "data/y_test.trc")

    x_test = torch.load("data/x_test.trc")
    x2_test = torch.load("data/x2_test.trc")
    y_test = torch.load("data/y_test.trc")

    idx = torch.randperm(x_test.size(0))
    x_test = x_test[idx]
    x2_test = x2_test[idx]
    y_test = y_test[idx]

    # filt = torch.all(x_test.reshape((-1, 1, 15*15)) != 0, dim=2)[:, 0]
    # x_test = x_test[filt]
    # x2_test = x2_test[filt]
    # y_test = y_test[filt]

    # x_test = torch.load("data/subset_x_test.trc")
    # x2_test = torch.load("data/subset_x2_test.trc")
    # y_test = torch.load("data/subset_y_test.trc")
    # x_train = torch.load("data/subset_x_train.trc")
    # x2_train = torch.load("data/subset_x2_train.trc")
    # y_train = torch.load("data/subset_y_train.trc")

    train_dataset = dataset(x_train, x2_train, y_train)
    test_dataset = dataset(x_test, x2_test, y_test)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    hmc = homemade_cnn(batch_size=batch_size, device=device).to(device)
    hmc = torch.load('NN_model/jumping-violet-236model.trc')
    opt = AdamW(hmc.parameters(), lr=lr)
    lambda1 = lambda epoch: 0.99 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
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
        scheduler.step()
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))
        wandb.log({'epoch': epoch, 'Learning rate': opt.param_groups[0]['lr']})

        if test_loss <= best_test_loss:
            bestmodel = copy.deepcopy(hmc)
            best_test_loss = test_loss
            bestmodel_epoch = epoch

        if epoch % 10 == 1:
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


def lol():
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(30, 50), layout='tight')
    for i in range(32):
        a = fig.add_subplot(8, 8, 2 * i + 1)
        imgplot = plt.imshow(y[0, i, :, :].cpu().detach().numpy())
        a.set_title('conv layer')
        a.axis("off")
        b = fig.add_subplot(8, 8, 2 * i + 2)
        plt.imshow(input[0, 0, :, :].reshape((15, 15)).cpu().detach().numpy())
        b.set_title('input')
    plt.show()


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
    model.eval()

    ddataset = dataset(conv, torch.tensor(dist), torch.tensor(dist))
    Data = DataLoader(ddataset, batch_size=100000)
    pred = torch.zeros_like(torch.tensor(dist))
    with torch.no_grad():
        for i, data in enumerate(Data):
            x_data, x2_data, y_data = data
            x_data, x2_data = x_data.to(device).type(torch.float), x2_data.to(device).type(torch.float)
            pred[i * 100000:(i + 1) * 100000] = model.forward(
                x_data[i * 100000:(i + 1) * 100000].reshape((-1, 1, 15, 15)),
                x2_data[i * 100000:(i + 1) * 100000].reshape((-1, 1)))
            torch.cuda.empty_cache()

    return pred


if __name__ == '__main__':
    nn_fid = train_generalized_CNN()
    from analyse_nn import analyse
    analyse(nn_fid)
