import numpy as np
import pandas as pd
import random
from tqdm import tqdm


class TrainDataSetup:
    def __init__(self, missing_percent, look_back, length_of_signal, missing_scope, epoch):
        self.missing_percent = missing_percent
        self.look_back = look_back
        self.length_of_signal = length_of_signal
        self.missing_scope = missing_scope
        self.epoch = epoch
        self.new_missing_percent = missing_percent

    def create_train_dataset(self, df):
        dataX1, dataX2, dataY1, dataY2 = [], [], [], []
        for i in tqdm(range(len(df) - self.look_back - 1)):
            dataX1.append(df.iloc[i: i + self.look_back, 0].values)
            dataX2.append(df.iloc[i: i + self.look_back, 1].values)
            dataY1.append(df.iloc[i + self.look_back, 0])
            dataY2.append(df.iloc[i + self.look_back, 1])
        return np.array(dataX1), np.array(dataX2), np.array(dataY1), np.array(dataY2)

    def blocks(self):
        blank_list = []
        i = 1
        while True:
            count_of_gaps = int((self.missing_percent / 10 * self.length_of_signal * 2) / (self.missing_scope * i))
            length_of_gap = (self.length_of_signal * 2 - (self.missing_scope * i) * count_of_gaps) / (count_of_gaps + 1)
            if length_of_gap >= 1:
                break
            else:
                i += 1
        length_of_gap = int(length_of_gap)

        for j in range(self.look_back + 1):
            blank_list.append(0)

        for _ in range(self.look_back + 1, self.length_of_signal * 2 - length_of_gap,
                       (self.missing_scope * i) + length_of_gap):
            for ms in range(length_of_gap):
                blank_list.append(0)
            for ms in range((self.missing_scope * i)):
                blank_list.append(1)
        for ms in range(length_of_gap):
            blank_list.append(0)
        self.new_missing_percent = np.round(sum(blank_list) / len(blank_list), 2) * 100
        return blank_list

    def random(self):
        blank_list = []
        blank_places = random.sample(range(self.length_of_signal*2),
                                     int(self.length_of_signal*2 * self.missing_percent / 10))
        for i in range(self.length_of_signal*2):
            if i in blank_places:
                blank_list.append(1)
            else:
                blank_list.append(0)
        return blank_list

    def test_data(self):
        if self.missing_scope != 0:
            return self.blocks()
        else:
            return self.random()

    def train_data(self, music1, music2):
        X0, X1, y0, y1 = self.create_train_dataset(pd.concat([music1.iloc[0: self.length_of_signal, :],
                                                              music2.iloc[0:self.length_of_signal, :]], axis=0))
        return (X0 + X1) // 2, (y0 + y1) // 2







