import torch.nn as nn

class TrainConfig():
    def set_data_folder(self, data_folder):
        self.data_folder = data_folder
        return self

    def set_num_threads(self, num_threads):
        self.num_threads = num_threads
        return self

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self
    
    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs
        return self
    
    def set_test_to_all_ratio(self, test_to_all_ratio):
        self.test_to_all_ratio = test_to_all_ratio
        return self

    def set_results_dir(self, results_dir):
        self.results_dir = results_dir
        return self

    def set_model(self, model):
        self.model = model
        return self

    def set_loss_criterion(self, loss_criterion):
        self.loss_criterion = loss_criterion
        return self

    def set_doa_classes(self, doa_classes):
        self.doa_classes = doa_classes
        return self

class Dropouts():
    def __init__(self, input_dropout, conv_dropout, lstm_dropout):
        self.input_dropout = nn.Dropout(input_dropout)
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)