import os
import numpy as np
import pysound as ps
import ir
import csv
from sklearn.model_selection import train_test_split
from batch_convolver import convolve
from feature_extractor import extract_feature
from model import CRNN, ConvNet
import torch
import torch.nn as nn
from config import TrainConfig, Dropouts, LSTM_FIRST, LSTM_FULL, LSTM_LAST
from doa_classes import DoaClasses
from shutil import copy2


def isolate_train_val(data_folder, output, ratio, seed, format='.wav'):
    labelpath = os.path.join(data_folder, 'labels.csv')
    csvfile = open(labelpath, 'r')
    csv_reader = csv.reader(csvfile, delimiter=',')
    header = next(csv_reader, None)
    dataset = []
    for line in csv_reader:
        npypath = os.path.join(data_folder, line[0])
        datapath = npypath.replace('.npy', format)
        if os.path.exists(datapath):
            dataset.append(line)

    train_data_entries, val_data_entries = train_test_split(dataset, test_size=ratio, random_state=seed)

    val_folder = os.path.join(output, 'val')
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    val_writer = csv.writer(open(os.path.join(val_folder, 'labels.csv'), 'w'))
    val_writer.writerow(header)
    for data in val_data_entries:
        old_path = os.path.join(data_folder, data[0]).replace('.npy', format)
        new_path = old_path.replace(data_folder, val_folder)
        copy2(old_path, new_path)
        val_writer.writerow(data)

    train_folder = os.path.join(output, 'train')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    train_writer = csv.writer(open(os.path.join(train_folder, 'labels.csv'), 'w'))
    train_writer.writerow(header)
    for data in train_data_entries:
        old_path = os.path.join(data_folder, data[0]).replace('.npy', format)
        new_path = old_path.replace(data_folder, train_folder)
        copy2(old_path, new_path)
        train_writer.writerow(data)



# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_model(formulation, model):
    dropouts = Dropouts(0, 0, 0)
    doa_classes = None
    if formulation == "Reg":
        loss = nn.MSELoss(reduction='sum')
        output_dimension = 3        
    elif formulation == "Class":
        loss = nn.CrossEntropyLoss(reduction="sum")
        doa_classes = DoaClasses()
        output_dimension = len(doa_classes.classes)

    if model == "CNN":
        model_choice = ConvNet(device, dropouts, output_dimension, doa_classes).to(device)
    elif model == "CRNN":
        model_choice = CRNN(device, dropouts, output_dimension, doa_classes, "Full").to(device)
    return model_choice
        

def save_rectangle_room(dims, filepath_out='rect.obj'):
    with open(filepath_out, 'w') as ofile:
        ofile.write("o Rect\n")
        for w in [0, dims[0]]:
            for l in [0, dims[1]]:
                for h in [0, dims[2]]:
                    line = "v {:.4f} {:.4f} {:.4f}\n".format(w, l, h)
                    ofile.write(line)
        ofile.write("vn -1.0000 0.0000 0.0000\n"
                    "vn 0.0000 1.0000 0.0000\n"
                    "vn 1.0000 0.0000 0.0000\n"
                    "vn 0.0000 -1.0000 0.0000\n"
                    "vn 0.0000 0.0000 -1.0000\n"
                    "vn 0.0000 0.0000 1.0000\n"
                    "f 2//1 3//1 1//1\n"
                    "f 4//2 7//2 3//2\n"
                    "f 8//3 5//3 7//3\n"
                    "f 6//4 1//4 5//4\n"
                    "f 7//5 1//5 3//5\n"
                    "f 4//6 6//6 8//6\n"
                    "f 2//1 4//1 3//1\n"
                    "f 4//2 8//2 7//2\n"
                    "f 8//3 6//3 5//3\n"
                    "f 6//4 2//4 1//4\n"
                    "f 7//5 5//5 1//5\n"
                    "f 4//6 2//6 6//6")
    ofile.close()


def save_feature(meshpath, speechpath, src_coord, lis_coord, absorb=0.1):
    try:
        mesh = ps.loadobj(meshpath, os.path.join(os.path.dirname(meshpath), ''), absorb)
        scene = ps.Scene()

        scene.setMesh(mesh)
        src = ps.Source(src_coord)
        src.radius = 0.01
        lis = ps.Listener(lis_coord)
        lis.radius = 0.01
        lis.channel_layout_type = ps.ChannelLayoutType.ambisonic

        res = ir.MultiSoundBuffer(**scene.computeMultichannelIR(src, lis))
        wavname = meshpath.replace('.obj', '.wav')
        if res.get_length() > 0.05:
            res.save(wavname)
        else:
            print("invalid IR, abort!")
            return None
        convolved_path = meshpath.replace('.obj', '_conv.wav')
        convolve(wavname, speechpath, convolved_path)
        feature_path = convolved_path.replace('.wav', '.npy')
        extract_feature(convolved_path, feature_path)
        return feature_path
    except Exception as e:
        print(str(e))
        return None


if __name__ == "__main__":
    # save_rectangle_room([5.3,5.9,2.38])
    # model = create_model("Reg", "CNN")
    #feature_path = save_feature('data/room.obj', 'data/84-121123-0000.flac', [1,1,1], [3,5,4])
    #if feature_path:
    #    features = np.load(feature_path)
