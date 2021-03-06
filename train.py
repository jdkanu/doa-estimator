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
from model import CRNN, ConvNet
from config import TrainConfig, Dropouts, LSTM_FIRST, LSTM_FULL, LSTM_LAST
from doa_classes import DoaClasses

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def TensorAngles(a, b):
    inner_product = (a * b).sum(dim=1)
    a_norm = a.pow(2).sum(dim=1).pow(0.5)
    b_norm = b.pow(2).sum(dim=1).pow(0.5)
    cos = inner_product / (a_norm * b_norm)
    angle = torch.acos(cos)
    angle[torch.isnan(angle)] = 0
    return angle

def output_has_all_frames(config):
    return config.lstm_output == LSTM_FULL and isinstance(config.model, CRNN)

# dataset
class CustomDataset(Dataset):
    def __init__(self, data_entries, train_config):
        self.len = len(data_entries)
        self.internal_data = data_entries
        self.train_config = train_config

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        try:
            data = np.load(self.internal_data[index][0])
        except Exception as e:
            print(str(e))
            print("Error loading: " + str(index))
        data = np.moveaxis(data, -1, 0)

        label = np.array(self.internal_data[index][1:4]).astype("float32")[0]
        if self.train_config.doa_classes:
            label = self.train_config.doa_classes.index_for_xyz(label)
        if output_has_all_frames(self.train_config):
            label = np.array([label]*25)

        return data, label


def doa_train(config):

    # initialize dataset
    if not os.path.exists(config.data_folder):
        print("data folder non-exist")
        return 100
    
    labelpath = os.path.join(config.data_folder, 'labels.csv')
    csvfile = open(labelpath, 'r')
    csv_reader = csv.reader(csvfile, delimiter=',')
    next(csv_reader, None)
    dataset = []
    for line in csv_reader:
        npypath = os.path.join(config.data_folder, line[0])
        if os.path.exists(npypath):
            dataset.append([npypath, [float(x) for x in line[1:4]]])
        # if len(dataset)>1000:
        #     break

    train_data_entries, val_data_entries = train_test_split(dataset, test_size=config.test_to_all_ratio, random_state=11)
    train_dataset = CustomDataset(train_data_entries, config)
    val_dataset = CustomDataset(val_data_entries, config)

    # initialize Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=False, num_workers=0)

    # Loss and optimizer
    criterion = config.loss_criterion
    optimizer = torch.optim.Adam(config.model.parameters(), lr=config.learning_rate)
    # optimizer = torch.optim.Adadelta(config.model.parameters(), lr=learning_rate)

    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
    writer = SummaryWriter(config.results_dir)
    angle_observations = np.array([5, 10, 15])

    # Train the model
    ts = time.time()
    num_iterations_before_early_stop = 3
    early_stop_flag = False
    early_stop_cnt = 0
    lowest_error = 1e6
    total_step = len(train_loader)
    for epoch in range(config.num_epochs):
        if not early_stop_flag:
            config.model.train()
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                images = images.float().to(device)
                if config.doa_classes:
                    labels = labels.long() 
                else:
                    labels = labels.float()
                labels = labels.to(device)
                outputs = config.model(images)
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
                          .format(epoch + 1, config.num_epochs, i + 1, total_step, loss.item() / len(labels), time.time()-ts))

            # Use val set to test the model at each epoch
            config.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            with torch.no_grad():
                total_val_loss = 0
                total_labels = 0
                angle_cnts = np.zeros(shape=angle_observations.shape)
                for images, labels in val_loader:
                    images = images.float().to(device)
                    if config.doa_classes:
                        labels = labels.long()
                    else:
                        labels = labels.float()
                    labels = labels.to(device)
                    outputs = config.model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    total_labels += len(labels)
                    if config.doa_classes:
                        if output_has_all_frames(config):
                            outputs = torch.sum(outputs, 2)
                            labels = labels[:, 0] # Can take the 0th b/c labels identical for frames
                        _, predicted = torch.max(outputs, 1)
                        P = [config.doa_classes.classes[x].get_xyz_vector() for x in predicted]
                        L = [config.doa_classes.classes[x].get_xyz_vector() for x in labels]
                        angles = TensorAngles(torch.tensor(P), torch.tensor(L))
                    else:
                        if output_has_all_frames(config):
                            outputs = torch.sum(outputs, 1)/25
                            labels = labels[:, 0]
                        angles = TensorAngles(outputs, labels)
                    for i, deg in enumerate(angle_observations):
                        angle_cnts[i] += (angles <= np.deg2rad(deg)).sum().item()
                average_val_loss = total_val_loss / total_labels
                writer.add_scalar("val_loss", average_val_loss, epoch)
                print('[Validation] Test Accuracy of the model at Epoch {}: {:.8f}'.format(epoch + 1, average_val_loss))
                angle_accuracy = angle_cnts / total_labels
                for i, accuracy in enumerate(angle_accuracy):
                    writer.add_scalar("deg{}_accuracy".format(angle_observations[i]), accuracy, epoch)
                print('             Anglular accuracy for {}:{}'.format(angle_observations, angle_accuracy))

            # remember lowest error
            if average_val_loss > lowest_error * 0.95:
                early_stop_cnt += 1
                if early_stop_cnt >= num_iterations_before_early_stop:
                    early_stop_flag = True
            if average_val_loss < lowest_error:
                lowest_error = average_val_loss
                early_stop_cnt = 0
                torch.save(config.model.state_dict(), os.path.join(config.results_dir, "best_valid.pth"))
        else:
            print("=> early stop with val error {:.8f}".format(lowest_error))
            writer.close()
            break  # early stop break
    return lowest_error


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser(prog='train',
                                     description="""Script to train the DOA estimation system""")
    parser.add_argument("--input", "-i", default="data", help="Directory where data and labels are", type=str)
    parser.add_argument("--savedir", "-s", default=".", help="Directory to write results", type=str)
    parser.add_argument("--rate", "-r", type=float, default=None, help="Choose a learning rate, default to sweep")
    parser.add_argument("--batchsize", "-b", type=int, default=None, help="Choose a batchsize, default to sweep")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--dropout", "-dp", type=float, default=0., help="Specify dropout rate")
    # parser.add_argument("--input_dropout", "-id", type=float, default=0., help="Specify input dropout rate")
    # parser.add_argument("--conv_dropout", "-cd", type=float, default=0., help="Specify conv dropout rate (applied at all layers)")
    # parser.add_argument("--lstm_dropout", "-ld", type=float, default=0., help="Specify lstm dropout rate (applied to lstm output)")
    parser.add_argument("--model", "-m", type=str, choices=["CNN", "CRNN"], required=True, help="Choose network model")
    parser.add_argument("--outputformulation", "-of", type=str, choices=["Reg", "Class"], required=True, help="Choose output formulation")
    parser.add_argument("--lstmout", "-lo", type=str, choices=[LSTM_FULL, LSTM_FIRST, LSTM_LAST], required=False, default=LSTM_FULL, help="Choose what to use from LSTM ouput")
    args = parser.parse_args()

    # dropouts = Dropouts(args.input_dropout, args.conv_dropout, args.lstm_dropout)
    dropouts = Dropouts(args.dropout, args.dropout, args.dropout)
    rates = [1e-5, 1e-7, 1e-3, 1e-9, 1e-1] if not args.rate else [args.rate]
    batches = [128, 32, 64] if not args.batchsize else [args.batchsize]

    for learning_rate in rates:
        for batch_size in batches:
            # dir to store the experiment files
            results_dir = os.path.join(args.savedir, \
                "results" + '_{}'.format(args.model) + '_{}'.format(args.outputformulation) + \
                '_lr{}'.format(learning_rate) + '_bs{}'.format(batch_size) + '_drop{}'.format(args.dropout))
            print('writing results to {}'.format(results_dir))

            doa_classes = None
            if args.outputformulation == "Reg":
                loss = nn.MSELoss(reduction='sum')
                output_dimension = 3
            elif args.outputformulation == "Class":
                loss = nn.CrossEntropyLoss(reduction="sum")
                doa_classes = DoaClasses(doa_grid_resolution = np.pi/12)
                output_dimension = len(doa_classes.classes)

            if args.model == "CNN":
                model_choice = ConvNet(device, dropouts, output_dimension, doa_classes).to(device)
            elif args.model == "CRNN":
                model_choice = CRNN(device, dropouts, output_dimension, doa_classes, args.lstmout).to(device)

            config = TrainConfig() \
                        .set_data_folder(args.input) \
                        .set_learning_rate(learning_rate) \
                        .set_batch_size(batch_size) \
                        .set_num_epochs(args.epochs) \
                        .set_test_to_all_ratio(0.1) \
                        .set_results_dir(results_dir) \
                        .set_model(model_choice) \
                        .set_loss_criterion(loss) \
                        .set_doa_classes(doa_classes) \
                        .set_lstm_output(args.lstmout)
            doa_train(config)
