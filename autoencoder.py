import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pickle
import wandb
import copy
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, device, grid_shape):
        super(Autoencoder, self).__init__()

        self.device = device
        self.grid_shape = grid_shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=int(self.grid_shape*3/10), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=int(self.grid_shape*3/10), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=int(self.grid_shape*3/10), stride=1, padding=0),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=int(self.grid_shape*3/10), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=int(self.grid_shape*3/10), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=int(self.grid_shape*3/10), stride=1, padding=0)
        )


    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.forward_encoder(x.reshape((-1, 1, self.grid_shape, self.grid_shape)))
        x = self.forward_decoder(x).reshape((-1, 1, self.grid_shape, self.grid_shape))
        return x


    def train_loop(self, Data, loss_fn, optimizer, epoch):
        Loss = 0
        test = 0
        for data in Data:
            optimizer.zero_grad()
            x_data, y_data = data
            mask = x_data[:, 1].reshape((-1, 1, self.grid_shape, self.grid_shape)).to(self.device)
            x_data = x_data[:, 0].to(self.device).reshape((-1, 1, self.grid_shape, self.grid_shape))
            pred = self.forward(x_data)
            pred = pred*mask
            loss = loss_fn(pred, x_data)
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]

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
            x_data, y_data = data
            mask = x_data[:, 1].reshape((-1, 1, self.grid_shape, self.grid_shape)).to(self.device)
            x_data = x_data[:, 0].to(self.device).reshape((-1, 1, self.grid_shape, self.grid_shape))
            pred = self.forward(x_data)
            pred = pred*mask
            loss = loss_fn(pred, x_data)
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]
            if log:
                wandb.log({"Test loss": loss.item(), "epoch": epoch})

        Loss = Loss / test

        return Loss

class MLP(nn.Module):
    def __init__(self, device, shape):
        super(MLP, self).__init__()

        self.device = device
        self.shape = shape

        self.mlp = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=int(self.shape * 3 / 10), stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*2*2, 1)
        )

    def forward(self, x):
        return self.mlp(x).reshape(-1)

    def train_loop(self, Data, loss_fn, optimizer, epoch):
        Loss = 0
        test = 0
        for data in Data:
            optimizer.zero_grad()
            x_data, y_data = data
            x_data, y_data = x_data.to(self.device), y_data.to(self.device)
            pred = self.forward(x_data)
            loss = loss_fn(pred, y_data)
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]

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
            x_data, y_data = data
            x_data, y_data = x_data.to(self.device), y_data.to(self.device)
            pred = self.forward(x_data)
            loss = loss_fn(pred, y_data)
            Loss += loss.item() * x_data.shape[0]
            test += x_data.shape[0]
            if log:
                wandb.log({"Test loss": loss.item(), "epoch": epoch})

        Loss = Loss / test

        return Loss


def filter_data(mode, shape=10):

    dict_conv = {"train": f"data/master_conv_with_mean_{shape}.trc", "test": f"data/test_master_conv_with_mean_{shape}.trc"}

    dict_ave = {"train": f"temp/master_ave_dist_list_{shape}.pkl", "test": f"temp/test_master_ave_dist_list_{shape}.pkl"}

    dict_dist = {"train": f"temp/master_scan_dist_list_{shape}.pkl", "test": f"temp/test_master_scan_dist_list_{shape}.pkl"}

    dict_save = {"train": [f"data/x_train_{shape}.trc", f"data/x2_train_{shape}.trc", f"data/y_train_{shape}.trc"],
                 "test": [f"data/x_test_{shape}.trc", f"data/x2_test_{shape}.trc", f"data/y_test_{shape}.trc"]}

    l_scan_case_dist = torch.load(dict_conv[mode])

    with open(dict_ave[mode], 'rb') as f:
        ave_dist = pickle.load(f)
    for i in range(ave_dist.__len__()):
        if i == 0:
            temp = ave_dist[i].reshape(-1)
        else:
            temp = np.concatenate((temp, ave_dist[i].reshape(-1)), axis=0)
    ave_dist = temp
    ave_dist = torch.from_numpy(np.array(ave_dist))

    with open(dict_dist[mode], 'rb') as f:
        center_dist = pickle.load(f)
    for i in range(center_dist.__len__()):
        if i == 0:
            temp = center_dist[i].reshape(-1)
        else:
            temp = np.concatenate((temp, center_dist[i].reshape(-1)), axis=0)
    center_dist = temp
    center_dist = torch.from_numpy(np.array(center_dist))

    ind = torch.where(center_dist != 0)[0]
    x_train = l_scan_case_dist[ind]
    x2_train = center_dist[ind]
    y_train = ave_dist[ind]

    x_train = x_train.reshape((-1, 2, shape, shape))
    sum = x_train.sum(axis=(2, 3))[:, 0]
    train_filt_max = np.percentile(sum, 99)
    train_filt_min = np.percentile(sum, 1)
    filt1 = (sum <= train_filt_max)
    filt2 = (sum >= train_filt_min)
    filt = (filt1) & (filt2)
    x_train = x_train[torch.nonzero(filt)][:, 0]
    x2_train = x2_train[torch.nonzero(filt)][:, 0]
    y_train = y_train[torch.nonzero(filt)][:, 0]

    diff = x2_train - y_train
    train_filt_max = np.percentile(diff, 99)
    train_filt_min = np.percentile(diff, 1)
    filt1 = (diff <= train_filt_max)
    filt2 = (diff >= train_filt_min)
    filt = (filt1) & (filt2)
    x_train = x_train[torch.nonzero(filt)[:, 0]]
    x2_train = x2_train[torch.nonzero(filt)[:, 0]]
    y_train = y_train[torch.nonzero(filt)[:, 0]]

    filt = torch.where(x_train[:, 0, :, :].sum(axis=(1,2)) != 0)[0]
    x_train = x_train[filt]
    x2_train = x2_train[filt]
    y_train = y_train[filt]

    batch = 1000
    sum = 0
    for i in tqdm(range(int(np.ceil(x_train.shape[0]) / batch))):
        x_train[i * batch:(i + 1) * batch, 0][x_train[i * batch:(i + 1) * batch, 0] != 0] += 0.5

    torch.save(x_train, dict_save[mode][0])
    torch.save(x2_train, dict_save[mode][1])
    torch.save(y_train, dict_save[mode][2])

    x_train = torch.load(dict_save[mode][0]).float()
    x2_train = torch.load(dict_save[mode][1]).float()
    y_train = torch.load(dict_save[mode][2]).float()

    torch.manual_seed(42)
    idx = torch.randperm(x_train.size(0))
    x_train = x_train[idx]
    x2_train = x2_train[idx]
    y_train = y_train[idx]

    return x_train, x2_train, y_train


class dataset(torch.utils.data.IterableDataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __iter__(self):
        return zip(self.X.float(), self.Y.float())

    def __len__(self):
        return len(self.X)


def train_mlp(encoder_fid):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epoch = 100
    batch_size = 10000
    shape = 10
    # Create the autoencoder model
    model = torch.load(encoder_fid).to(device)

    # Create the MLP model
    mlp = MLP(device, shape).to(device)

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.0001)
    lambda1 = lambda epoch: 0.995 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # === train data import ===
    x_train, x2_train, y_train = filter_data("train", shape)
    # enc_x_train = torch.zeros((x_train.shape[0], 16, int(shape-6), int(shape-6)))
    # with torch.no_grad():
    #     for i in range(int(np.ceil(x_train.shape[0] / 10000))):
    #         enc_x_train[i * 10000:(i + 1) * 10000] = model.forward_encoder(x_train[i * 10000:(i + 1) * 10000, 0].to(device).reshape((-1, 1, shape, shape))).reshape(
    #             -1, 16, int(shape-6), int(shape-6))
    #         torch.cuda.empty_cache()
    #     torch.save(enc_x_train, "data/enc_x_train.trc")
    enc_x_train = torch.load("data/enc_x_train.trc")

    # === Test data import ===
    x_test, x2_test, y_test = filter_data("test")
    # enc_x_test = torch.zeros((x_test.shape[0], 16, int(shape-6), int(shape-6)))
    # with torch.no_grad():
    #     for i in range(int(np.ceil(x_test.shape[0] / 10000))):
    #         enc_x_test[i * 10000:(i + 1) * 10000] = model.forward_encoder(x_test[i * 10000:(i + 1) * 10000, 0].to(device).reshape((-1, 1, shape, shape))).reshape(
    #             -1, 16, int(shape-6), int(shape-6))
    #         torch.cuda.empty_cache()
    #     torch.save(enc_x_test, "data/enc_x_test.trc")
    enc_x_test = torch.load("data/enc_x_test.trc")

    train_dataset = dataset(enc_x_train, y_train)
    test_dataset = dataset(enc_x_test, y_test)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    wandb.init(project='encoded_MLP', mode='online')
    if wandb.run.name is None:
        wandb.run.name = 'offline_test'
    wandb.watch(mlp, log_freq=10)
    test_loss = np.inf
    best_test_loss = np.inf
    pbar = tqdm(range(max_epoch), desc="test loss = " + str(test_loss) + " Train_loss = " + str(test_loss))
    for epoch in pbar:
        mlp.train()
        train_loss = mlp.train_loop(train_data, criterion, optimizer, epoch)
        wandb.log({"Mean train loss": train_loss, "epoch": epoch})
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))

        mlp.eval()
        test_loss = mlp.test_loop(test_data, criterion, epoch)
        wandb.log({"Mean test loss": test_loss, "epoch": epoch})
        scheduler.step()
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))
        wandb.log({'epoch': epoch, 'Learning rate': optimizer.param_groups[0]['lr']})

        if test_loss <= best_test_loss:
            bestmodel = copy.deepcopy(model)
            best_test_loss = test_loss
            bestmodel_epoch = epoch

        if epoch % 10 == 1:
            torch.save(bestmodel, "NN_model/" + wandb.run.name + 'model.trc')
            print('saved best model with loss ' + str(best_test_loss) + ' at epoch +' + str(bestmodel_epoch))


def train_autoencoder():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epoch = 100
    batch_size = 1000
    shape = 10
    # Create the autoencoder model
    model = Autoencoder(device, shape).to(device)

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lambda1 = lambda epoch: 0.995 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # === train data import ===
    print("loading train data")
    x_train, x2_train, y_train = filter_data("train", shape)

    # === Test data import ===
    print("loading test data")
    x_test, x2_test, y_test = filter_data("test", shape)

    train_dataset = dataset(x_train, x_train)
    test_dataset = dataset(x_test, x_test)

    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)

    wandb.init(project='Autoencoder', mode='online')
    if wandb.run.name is None:
        wandb.run.name = 'offline_test'
    wandb.watch(model, log_freq=10)
    test_loss = np.inf
    best_test_loss = np.inf
    pbar = tqdm(range(max_epoch), desc="test loss = " + str(test_loss) + " Train_loss = " + str(test_loss))
    for epoch in pbar:
        model.train()
        train_loss = model.train_loop(train_data, criterion, optimizer, epoch)
        wandb.log({"Mean train loss": train_loss, "epoch": epoch})
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))

        model.eval()
        test_loss = model.test_loop(test_data, criterion, epoch)
        wandb.log({"Mean test loss": test_loss, "epoch": epoch})
        scheduler.step()
        pbar.set_description(desc="test loss = " + str(test_loss) + " Train_loss = " + str(train_loss))
        wandb.log({'epoch': epoch, 'Learning rate': optimizer.param_groups[0]['lr']})

        if test_loss <= best_test_loss:
            bestmodel = copy.deepcopy(model)
            best_test_loss = test_loss
            bestmodel_epoch = epoch

        if epoch % 10 == 1:
            torch.save(bestmodel, "NN_model/" + wandb.run.name + 'model.trc')
            print('saved best model with loss ' + str(best_test_loss) + ' at epoch +' + str(bestmodel_epoch))
            # train_dataset.current_subset_size += int(np.round(len(x_train)*0.1))
            # test_dataset.current_subset_size += int(np.round(len(x_test)*0.1))

    return f"NN_model/{wandb.run.name}model.trc"


if __name__ == '__main__':
    encoder_fid = train_autoencoder()
    train_mlp(encoder_fid)
