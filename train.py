import os
import sys
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from tensorboardX import SummaryWriter
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from itertools import compress
import argparse

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# dataset
class CustomDataset(Dataset):
    def __init__(self, data_entries):
        self.len = len(data_entries)
        self.internal_data = data_entries
        # self.data_rand = np.random.uniform(0, 1, [6, 25, 513]).astype('f')
        # self.label_rand = np.random.uniform(0, 1, 3).astype('f')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        try:
            data = np.load(self.internal_data[index][0])
        except Exception as e:
            print(str(e))
            print("Error loading: " + str(index))
        data = np.moveaxis(data, -1, 0)
        label = np.array(self.internal_data[index][1:4]).astype("float32")

        return data, label[0]

class ConvNet(nn.Module):
    def __init__(self, dropouts):
        super(ConvNet, self).__init__()
        self.dropouts = dropouts
        conv_layers = [
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (bsz, 64, 25, 64)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),  # (bsz, 64, 25, 8)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),  # (bsz, 64, 25, 2)
        ]
        modules = []
        for layer in conv_layers:
            modules.append(layer)
            modules.append(dropouts.conv_dropout)
        self.conv = nn.Sequential(*modules)

        # from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
        self.hidden_size = 64
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(self.hidden_size*2, 3)

    def forward(self, x):
        out = self.dropouts.input_dropout(x)
        out = self.conv(x)  # (bsz, 64, 25, 2)
        reshape = out.permute(0, 2, 1, 3).contiguous().view(len(out), 25, 128)
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, reshape.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, reshape.size(0), self.hidden_size).to(device)

        lstm_out, _ = self.lstm(reshape, (h0, c0))
        fc_out = self.fc(lstm_out[:, -1, :]) # NOTE: revisit to use more than just the last LSTM output

        return fc_out


def diffraction_train(num_epochs, batch_size, learning_rate, dropouts, TEST_TO_ALL_RATIO, data_folder, results_dir, nthreads=8):

    # initialize dataset
    if not os.path.exists(data_folder):
        print("data folder non-exist")
        return
    
    labelpath = os.path.join(data_folder, 'labels.csv')
    csvfile = open(labelpath, 'r')
    csv_reader = csv.reader(csvfile, delimiter=',')
    next(csv_reader, None)
    dataset = []
    for line in csv_reader:
        npypath = os.path.join(data_folder, line[0])
        # if os.path.exists(npypath):
        dataset.append([npypath, [float(x) for x in line[1:4]]])
        # if len(dataset)>1000:
        #     break

    train_data_entries, val_data_entries = train_test_split(dataset, test_size=TEST_TO_ALL_RATIO)
    train_dataset = CustomDataset(train_data_entries)
    val_dataset = CustomDataset(val_data_entries)

    # initialize Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=nthreads)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=nthreads)

    model = ConvNet(dropouts).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    os.makedirs(results_dir)
    writer = SummaryWriter(results_dir)

    # Train the model
    ts = time.time()
    num_iterations_before_early_stop = 1
    early_stop_flag = False
    early_stop_cnt = 0
    lowest_error = 1e6
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        if not early_stop_flag:
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                images = images.float().to(device)
                labels = labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log
                niter = epoch * len(train_loader) + i+1
                writer.add_scalar("training_loss", loss.item() / len(labels), niter)
                if i % 50 == 0:
                    print('[Training] Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}, time elapsed: {:.2f} seconds'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item() / len(labels), time.time()-ts))

            # Use val set to test the model at each epoch
            model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            with torch.no_grad():
                total_val_loss = 0
                total_labels = 0
                for images, labels in val_loader:
                    images = images.float().to(device)
                    labels = labels.float().to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss
                    total_labels += len(labels)
                average_val_loss = total_val_loss.item() / total_labels
                writer.add_scalar("val_loss", average_val_loss, epoch)
                print('[Validation] Test Accuracy of the model at Epoch {}: {:.8f}'.format(epoch + 1, average_val_loss))

            # remember lowest error
            if average_val_loss < lowest_error:
                lowest_error = average_val_loss
                early_stop_cnt = 0
                torch.save(model.state_dict(), os.path.join(results_dir, "best_valid.pth"))
            else:
                early_stop_cnt += 1
                if early_stop_cnt >= num_iterations_before_early_stop:
                    early_stop_flag = True
        else:
            print("=> early stop with val error {:.8f}".format(lowest_error))
            writer.close()
            break  # early stop break

class Dropouts():
    def __init__(self, input_dropout, conv_dropout, lstm_dropout):
        self.input_dropout = nn.Dropout(input_dropout)
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='diffraction_CNNtrain',
                                     description="""Script to train the InstantDiffraction system""")
    parser.add_argument("--input", "-i", required=False, default="data", help="Directory where data and labels are", type=str)
    parser.add_argument("--nthreads", "-n", type=int, default=0, help="Number of threads to use")
    parser.add_argument("--rate", "-r", type=float, default=-1, help="Choose a learning rate, default to sweep")
    parser.add_argument("--batchsize", "-b", type=int, default=-1, help="Choose a batchsize, default to sweep")
    parser.add_argument("--input_dropout", "-id", type=float, default=0.1, help="Specify input dropout rate")
    parser.add_argument("--conv_dropout", "-cd", type=float, default=0.1, help="Specify conv dropout rate (applied at all layers)")
    parser.add_argument("--lstm_dropout", "-ld", type=float, default=0.1, help="Specify conv dropout rate (applied to lstm output)")
    args = parser.parse_args()

    # Hyper parameters
    num_epochs = 5
    TEST_TO_ALL_RATIO = 0.01
    data_folder = args.input
    nworkers = args.nthreads
    rate_select = args.rate
    batch_select = args.batchsize
    dropouts = Dropouts(args.input_dropout, args.conv_dropout, args.lstm_dropout)

    rates = [1e-3, 1e-9, 1e-5, 1e-7]
    batches = [32, 64, 128, 256]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if rate_select > 0:
        rates = [rate_select]
    if batch_select > 0:
        batches = [batch_select]
    for learning_rate in rates:
        for batch_size in batches:
            # dir to store the experiment files
            results_dir = "results/results" + time.strftime("_%Y_%m_%d_%H_%M_%S") + '_lr{}'.format(learning_rate) + '_bs{}'.format(batch_size)
            print('writing results to {}'.format(results_dir))
            diffraction_train(num_epochs, batch_size, learning_rate, dropouts, TEST_TO_ALL_RATIO, data_folder, results_dir, nthreads=nworkers)
