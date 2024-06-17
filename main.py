import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from torchsummary import summary

import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


class MetaNet(torch.nn.Module):

    def __init__(self, drop, out_numb):
        super(MetaNet, self).__init__()
        self.conv1 = torch.nn.Conv3d(3, 6, (25, 25, 3), padding='same', dtype=torch.float64)
        self.conv2 = torch.nn.Conv3d(6, 16, (17, 17, 3), padding='same', dtype=torch.float64)
        self.conv3 = torch.nn.Conv3d(16, 32, (8, 8, 3), padding='same', dtype=torch.float64)
        self.conv4 = torch.nn.Conv3d(32, 64, (5, 5, 3), padding='same', dtype=torch.float64)
        self.fc1 = torch.nn.Linear(1536, 512, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(512, 128, dtype=torch.float64)
        self.out = torch.nn.Linear(128, out_numb, dtype=torch.float64)  # appr - 11, inter - 15, dots - 40
        self.drop = torch.nn.Dropout(p=drop, inplace=False)
        self.in1 = torch.nn.InstanceNorm3d(3)
        self.in2 = torch.nn.InstanceNorm3d(16)
        self.in3 = torch.nn.InstanceNorm3d(64)

    def forward(self, x):
        x = self.in1(x)
        x = self.conv2(self.conv1(x))
        x = self.in2(x)
        x = F.max_pool3d(F.relu(x), (10, 10, 1))
        x = self.conv4(self.conv3(x))
        x = self.in3(x)
        x = F.max_pool3d(F.relu(x), (10, 10, 1))
        x = x.view(-1, self.num_flat_features(x))
        x = self.drop(x)
        x = F.relu(self.fc2(self.fc1(x)))
        x = self.out(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_data(files, device):
    structs = []
    coefs = []

    # df = pd.read_csv('appr_top.csv', header=None)
    df = pd.read_csv('points_top.csv', header=None)
    arr = np.array(df)
    i = 0
    for file in files:
        data = np.load(file, allow_pickle=True)
        if len(data['coef']) > 19:
            new_data_labels = data['coef'][0::2]
        else:
            new_data_labels = data['coef']
        new_data_labels = data['coef']
        # print(new_data_labels)
        # new_data_labels = arr[i].reshape(-1, 1)
        # new_data_labels = arr[i]
        # inputs, labels = torch.from_numpy(data["struct"]).permute(3, 0, 1, 2).unsqueeze(0).float().to(
        #     device), torch.from_numpy(data["coef"]).unsqueeze(0).float().to(device)
        inputs, labels = torch.from_numpy(data["struct"]).permute(3, 0, 1, 2).unsqueeze(0).float().to(
            device), torch.from_numpy(new_data_labels).unsqueeze(0).float().to(device)
        inputs = F.interpolate(inputs, size=(256, 256, 6))
        i += 1
        structs.append(inputs)
        coefs.append(labels)
    return torch.cat(structs), torch.cat(coefs)


files = sorted(glob("processed_data/points/Figure*.npz"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs, labels = get_data(files, device)


# print(labels)


def main():
    tipo = 'points'
    if tipo == 'points':
        out_numb = 40
    elif tipo == 'inter':
        out_numb = 15
    elif tipo == 'appr':
        out_numb = 11
    else:
        out_numb = 11

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MetaNet(drop=0.4, out_numb=out_numb).float().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.NAdam(net.parameters(), lr=0.001)
    torch.manual_seed(42)

    files = sorted(glob(f"processed_data/{tipo}/Figure*.npz"))
    inputs, labels = get_data(files, device)
    # (0, 1, 9) _____ (2, 3, 4, 5, 6, 7, 8) _____ (figure 1, 5: 0.2-2) (figure 2, 3, 4: 140 - 371)
    # perm = (6, 1, 2, 3, 4, 5, 7, 8, 9, 0)
    # perm = (0, 2, 3, 5, 6, 7, 1, 8, 4, 9)
    # perm = (9, 0, 1)
    perm = (2, 3, 6, 7, 5, 8, 4)

    best_dev_loss = 10000000000
    unsuc_epochs = 0
    train_losses = []
    val_losses = []
    tolerence = 23

    epochs = 15
    test_ = perm[0]

    mean = []
    for epoch in range(epochs):  # эпохи обучения
        running_loss = 0.0
        net.train()
        for input, label in zip(inputs[perm[2:], ...], labels[perm[2:], ...]):
            optimizer.zero_grad()
            output = net(input.unsqueeze(0))
            loss = criterion(output[0], torch.reshape(label, (1, out_numb))[0])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        net.eval()
        val_loss_ = 0
        for i in range(2):
            with torch.inference_mode():
                output = net(inputs[perm[i]].unsqueeze(0))[0]
                val_loss = criterion(output, torch.reshape(labels[perm[i]], (out_numb,)))
                val_loss_ += float(val_loss)
        val_loss_ = val_loss_ / 2
        val_loss = val_loss_
        val_losses.append(val_loss_)
        if val_loss < best_dev_loss:
            unsuc_epochs = 0
            best_dev_loss = val_loss
            print(f"saving best model with dev loss: {val_loss}")
            torch.save(net.state_dict(), 'net.pth')
        else:
            unsuc_epochs += 1
            if unsuc_epochs == tolerence:
                break
        train_losses.append(running_loss / 5)
        print(f"Epoch number: {epoch}, train loss: {running_loss / 5}, val_loss: {val_loss}")
        running_loss = 0.0

        # calculating general loss for epoch
        test_ = perm[0]
        outputs = net(inputs[test_].unsqueeze(0))[0]
        label_res = [round(val, 20) for val in labels[test_].detach().numpy().flatten()]
        new_out = []
        for val in outputs.detach().numpy():
            if val > 0:
                new_out.append(val)
            else:
                new_out.append(0)
        out_res = new_out
        new = np.array([list(label_res), out_res])
        new = new.T
        new = [sum(val) / len(val) for val in new]
        print(f"General loss for epoch {epoch}: ")
        print(sum(new) / len(new))
        print()
        mean.append(sum(new) / len(new))

    print('Finished Training')
    summary(net, (3, 256, 256, 6))
    print()

    # calculating mean loss for general loss
    x_loss = np.arange(1, epochs + 1, step=1)
    plt.plot(x_loss, np.array(train_losses))
    plt.plot(x_loss, np.array(val_losses))
    # plt.plot(x_loss, np.array(mean))
    plt.gca().legend(('Тренування', 'Апробація', 'Середня похибка'))
    plt.ylim(0, 1)
    plt.xlabel("Номер епохи")
    plt.ylabel("Величина середньоквадратичної похибки")
    plt.show()

    net.load_state_dict(torch.load('net.pth'))
    net.eval()

    # showing prediction results
    outputs = net(inputs[test_].unsqueeze(0))[0]
    label_res = [round(val, 20) for val in labels[test_].detach().numpy().flatten()]
    new_out = []
    for val in outputs.detach().numpy():
        if val > 0:
            new_out.append(val)
        else:
            new_out.append(0)
    out_res = new_out
    print("Реальні: ", label_res)
    print("Прогнозовані: ", out_res)

    # plots of prediction results
    if tipo == 'points':

        x_labels = label_res[:20]
        x_out = out_res[:20]
        plt.plot(x_labels, label_res[20:], 'b--')
        # plt.plot(x_out,  out_res[20:])
        plt.plot(x_labels, out_res[20:], 'r--')
        plt.gca().legend(('Реальні: ', 'Прогнозовані: '))
        plt.xlabel("Масштабована частота")
        plt.ylabel("Коефіцієнт пропускання")
        # plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
    else:
        x = np.arange(1, out_numb + 1, step=1)
        plt.plot(x, label_res, 'b--')
        plt.plot(x, out_res, 'r--')
        plt.gca().legend(('Реальні: ', 'Прогнозовані: '))
        plt.xlabel("Порядок коефіцієнту")
        plt.ylabel("Масштабована величина коефіцієнту")
        plt.ylim(0, 1)
        plt.xlim(10, 0)
        plt.show()

    make_dot(outputs, params=dict(net.named_parameters())).render('net', format='png')


main()
