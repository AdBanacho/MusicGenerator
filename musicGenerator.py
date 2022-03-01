from dataToFromFiles import load_data_from_npy, create_data_from_wav_to_npy
from models import training
from loadData import TrainDataSetup
from signalSave import predict_and_save_signals, original_signal_save, blank_signal_save


class MusicGenerator:
    def __init__(self, epochs, length_of_signal, missing_scope, look_back, missing_percent):
        self.X = 0
        self.y = 0
        self.blank_gaps = 0
        self.epochs = epochs
        self.train_data_setup = TrainDataSetup(missing_percent=missing_percent // 10,
                                               look_back=look_back,
                                               length_of_signal=length_of_signal,
                                               missing_scope=missing_scope,
                                               epoch=epochs)

    def create_data(self):
        create_data_from_wav_to_npy(self.train_data_setup)

    def load_data(self):
        self.X, self.y, self.blank_gaps = load_data_from_npy(self.train_data_setup)

    def train(self, ann, rnnR, rnnL):
        training(ann=ann, rnnR=rnnR, rnnL=rnnL,
                 X=self.X, y=self.y, epochs=self.epochs, train_data_set_up=self.train_data_setup)

    def predict(self, ann, rnnR, rnnL):
        predict_and_save_signals(ann=ann, rnnR=rnnR, rnnL=rnnL,
                                 original_signal=self.X,
                                 blank_gaps=self.blank_gaps,
                                 train_data_set_up=self.train_data_setup
                                 )
        blank_signal_save(train_data_set_up=self.train_data_setup,
                          original_signal=self.X,
                          blank_gaps=self.blank_gaps
                          )
        original_signal_save(original_signal=self.y, train_data_set_up=self.train_data_setup)
