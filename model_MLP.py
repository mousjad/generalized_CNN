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
from torchvision import transforms
from  utils import git_push

cudnn.benchmark = True
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(size=(10, 10)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=0.01)
])


class homemade_cnn(Module):
    def __init__(self, step=1, n_case=15, batch_size=2, device=torch.device("cpu")):
        super(homemade_cnn, self).__init__()
        self.step = step
        self.n_case = n_case
        self.batch_size = batch_size
        self.device = device
        self.dropout_rate = 0.5
        self.mask_max_pool = MaxPool2d(3, stride=1)
        self.mask_max_pool5 = MaxPool2d(5, stride=1)
        self.drop = Dropout(self.dropout_rate)
        # w1, w2, w3, w4 = wandb.config.w1, wandb.config.w2, wandb.config.w3, wandb.config.w4,
        w5, w6, w7, w8, w9 = wandb.config.w5, wandb.config.w6, wandb.config.w7, wandb.config.w8, wandb.config.w9
        w10 = wandb.config.w10
        self.Lin1 = Linear(w6, w7, bias=False)
        self.lr1 = ReLU()
        self.Lin2 = Linear(w7, w8, bias=False)
        self.lr2 = ReLU()
        self.Lin3 = Linear(w8, w9, bias=False)
        self.lr3 = ReLU()
        self.Lin4 = Linear(w9, w10, bias=False)
        self.Lin5 = Linear(w10, 1, bias=False)

    def forward(self, input, input2, in_training=False):
        # y = neighboorPadding(input[:, 0].reshape((-1, 1, 10, 10)), input[:, 1].reshape((-1, 1, 10, 10)), 3)
        y = input[:, 0].reshape((-1, 1, 10, 10))
        # y = data_transforms["train" if self.training else "val"](y)
        # y = self.drop1(self.norm1(self.r1(self.c1(y))))
        # mask = self.mask_max_pool(input[:, 1].reshape((-1, 1, 10, 10)))
        # y = y * mask
        #
        # y = self.drop2(self.norm2(self.r2(self.c2(y))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask
        #
        # y = self.drop3(self.norm3(self.r3(self.c3(y))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask
        #
        # y = self.drop4(self.norm4(self.r4(self.c4(y))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask

        # y = self.drop5(self.norm5(self.r5(self.p5(self.c5(y)))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask
        #
        # y = self.drop6(self.norm6(self.r6(self.p6(self.c6(y)))))
        # mask = self.mask_max_pool(mask)
        # y = y * mask
        # y = self.drop7(self.r7(self.c7(y)))
        # mask = self.mask_max_pool(mask)
        # y = y * mask
        # y = torch.flatten(self.drop8(self.r8(self.c8(y))), start_dim=1)
        # y = self.Lin1(y)
        y = torch.cat((torch.flatten(y, start_dim=1), input2[:, None]), 1)
        y = self.drop(self.self.self.lr1(self.Lin1(torch.flatten(y, start_dim=1))))
        y = self.drop(self.lr2(self.Lin2(y)))
        y = self.drop(self.lr3(self.Lin3(y)))
        y = self.drop(self.Lin4(y))
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
            y_data = y_data + (0.01 * torch.rand(y_data.shape[0]).to(self.device)-0.005)
            pred = self.forward(data_transforms(x_data), x2_data)
            loss = loss_fn(pred, y_data)
            Loss += loss.item() * x_data.shape[0]
            # Loss += (loss.item() - loss_fn(x2_data, y_data).item()) * x_data.shape[0]
            test += x_data.shape[0]
            l2 = torch.nn.functional.mse_loss(pred, y_data, reduction="none")
            wandb.log({"Train median": torch.median(l2), "epoch": epoch})

            # Backpropagation
            loss.backward()
            optimizer.step()
            wandb.log({"Train loss": loss.item(), "epoch": epoch})

        Loss = Loss / test

        return Loss


    def test_loop(self, Data, loss_fn, epoch, log=True):
        Loss = 0
        test = 0
        for data in Data:
            x_data, x2_data, y_data = data
            x_data, x2_data, y_data = x_data.to(self.device), x2_data.to(self.device), y_data.to(self.device)
            pred = self.forward(x_data, x2_data)
            loss = loss_fn(pred, y_data)
            Loss += loss.item() * x_data.shape[0]
            # Loss += (loss.item() - loss_fn(x2_data, y_data).item()) * x_data.shape[0]
            test += x_data.shape[0]
            if log:
                l2 = torch.nn.functional.mse_loss(pred, y_data, reduction="none")
                wandb.log({"Test median": torch.median(l2), "epoch": epoch})
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

def filter_data(mode):

    dict_conv = {"train": "data/master_conv_with_mean.trc", "test": "data/test_master_conv_with_mean.trc"}

    dict_ave = {"train": "temp/master_ave_dist_list.pkl", "test": "temp/test_master_ave_dist_list.pkl"}

    dict_dist = {"train": "temp/master_scan_dist_list.pkl", "test": "temp/test_master_scan_dist_list.pkl"}

    dict_save = {"train": ["data/x_train.trc", "data/x2_train.trc", "data/y_train.trc"],
                 "test": ["data/x_test.trc", "data/x2_test.trc", "data/y_test.trc"]}

    # l_scan_case_dist = torch.load(dict_conv[mode])
    #
    # with open(dict_ave[mode], 'rb') as f:
    #     ave_dist = pickle.load(f)
    # for i in range(ave_dist.__len__()):
    #     if i == 0:
    #         temp = ave_dist[i].reshape(-1)
    #     else:
    #         temp = np.concatenate((temp, ave_dist[i].reshape(-1)), axis=0)
    # ave_dist = temp
    # ave_dist = torch.from_numpy(np.array(ave_dist))
    #
    # with open(dict_dist[mode], 'rb') as f:
    #     center_dist = pickle.load(f)
    # for i in range(center_dist.__len__()):
    #     if i == 0:
    #         temp = center_dist[i].reshape(-1)
    #     else:
    #         temp = np.concatenate((temp, center_dist[i].reshape(-1)), axis=0)
    # center_dist = temp
    # center_dist = torch.from_numpy(np.array(center_dist))
    #
    # ind = torch.where(center_dist != 0)[0]
    # x_train = l_scan_case_dist[ind]
    # x2_train = center_dist[ind]
    # y_train = ave_dist[ind]
    #
    # x_train = x_train.reshape((-1, 2, 10, 10))
    # sum = x_train.sum(axis=(2, 3))[:, 0]
    # train_filt_max = np.percentile(sum, 99)
    # train_filt_min = np.percentile(sum, 1)
    # filt1 = (sum <= train_filt_max)
    # filt2 = (sum >= train_filt_min)
    # filt = (filt1) & (filt2)
    # x_train = x_train[torch.nonzero(filt)][:, 0]
    # x2_train = x2_train[torch.nonzero(filt)][:, 0]
    # y_train = y_train[torch.nonzero(filt)][:, 0]
    #
    # diff = x2_train - y_train
    # train_filt_max = np.percentile(diff, 99)
    # train_filt_min = np.percentile(diff, 1)
    # filt1 = (diff <= train_filt_max)
    # filt2 = (diff >= train_filt_min)
    # filt = (filt1) & (filt2)
    # x_train = x_train[torch.nonzero(filt)[:, 0]]
    # x2_train = x2_train[torch.nonzero(filt)[:, 0]]
    # y_train = y_train[torch.nonzero(filt)[:, 0]]
    #
    # filt = torch.where(x_train[:, 0, :, :].sum(axis=(1,2)) != 0)[0]
    # x_train = x_train[filt]
    # x2_train = x2_train[filt]
    # y_train = y_train[filt]
    #
    # torch.save(x_train, dict_save[mode][0])
    # torch.save(x2_train, dict_save[mode][1])
    # torch.save(y_train, dict_save[mode][2])

    x_train = torch.load(dict_save[mode][0]).float()
    x2_train = torch.load(dict_save[mode][1]).float()
    y_train = torch.load(dict_save[mode][2]).float()

    idx = torch.randperm(x_train.size(0))
    x_train = x_train[idx]
    x2_train = x2_train[idx]
    y_train = y_train[idx]

    return x_train, x2_train, y_train


def train_generalized_CNN():

    # === train data import ===
    x_train, x2_train, y_train = filter_data("train")

    # === Test data import ===
    x_test, x2_test, y_test = filter_data("test")

    hyperparameter_defaults = dict(
        batch_size=10000,
        lr=2e-5,
        epochs=2,
        w1=8,
        w2=16,
        w3=32,#
        w4=32,#
        w5=16,#
        w6=10*10 + 1,
        w7=64,
        w8=32,
        w9=16,
        w10=8
    )

    wandb.init(project='MLP on image', mode='online', config=hyperparameter_defaults)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    max_epoch = 100
    if wandb.run.name is None:
        wandb.run.name = 'offline_test'

    git_push(r"C:\Generalized_CNN\.git", f"{wandb.run.name}_automated_commit")
    l_fn = MSELoss(reduction='mean')

    train_dataset = dataset(x_train, x2_train, y_train)
    test_dataset = dataset(x_test, x2_test, y_test)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    hmc = homemade_cnn(batch_size=batch_size, device=device).to(device)
    # hmc = torch.load("NN_model/peachy-snowball-133model.trc")
    opt = Adam(hmc.parameters(), lr=lr)
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

    val_train_loss = bestmodel.test_loop(train_data, l_fn, epoch, log=False)
    val_test_loss = bestmodel.test_loop(test_data, l_fn, epoch)
    print('best model training loss: ' + str(val_train_loss))
    print('best model test loss: ' + str(val_test_loss))
    torch.cuda.empty_cache()
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
        plt.imshow(input[0, 0, :, :].reshape((10, 10)).cpu().detach().numpy())
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

            filt = torch.where(x_data != 0)
            x_data[filt] = x_data[filt] + 0.5
            x_mask = torch.zeros_like(x_data)
            x_mask[torch.where(x_data != 0)] = 1
            x_data = torch.cat((x_data.reshape((-1, 1, 10, 10)), x_mask.reshape((-1, 1, 10, 10))), dim=1)

            pred[i * 100000:(i + 1) * 100000] = model.forward(
                x_data[i * 100000:(i + 1) * 100000].reshape((-1, 2, 10, 10)),
                x2_data[i * 100000:(i + 1) * 100000].reshape((-1, 1)))
            torch.cuda.empty_cache()

    return pred


if __name__ == '__main__':
    nn_fid = train_generalized_CNN()
    # from analyse_nn import analyse
    # analyse(nn_fid)
