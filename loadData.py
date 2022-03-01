import numpy as np
import pandas as pd
from tqdm import tqdm


class TrainDataSetup:
    def __init__(self, missing_percent, look_back, length_of_signal, missing_scope, epoch):
        self.missing_percent = missing_percent
        self.look_back = look_back
        self.length_of_signal = length_of_signal
        self.missing_scope = missing_scope
        self.epoch = epoch

    def create_train_dataset(self, df):
        dataX1, dataX2, dataY1, dataY2 = [], [], [], []
        for i in tqdm(range(len(df) - self.look_back - 1)):
            dataX1.append(df.iloc[i: i + self.look_back, 0].values)
            dataX2.append(df.iloc[i: i + self.look_back, 1].values)
            dataY1.append(df.iloc[i + self.look_back, 0])
            dataY2.append(df.iloc[i + self.look_back, 1])
        return np.array(dataX1), np.array(dataX2), np.array(dataY1), np.array(dataY2)

    def list_to_remove(self, end, length_of_gaps, count_of_remove_scopes):
        lista = []
        for i in range(count_of_remove_scopes):
            g1 = end - length_of_gaps * i - self.missing_scope * (i + 1)
            g2 = end - length_of_gaps * i - self.missing_scope * i
            for j in range(g1, g2):
                lista.append(j)
        return lista

    def list_of_blank_gaps(self, elements_to_remove):
        list_of_missing_elements = []
        for i in tqdm(range(self.length_of_signal * 2)):
            if i in elements_to_remove:
                list_of_missing_elements.append(1)
            else:
                list_of_missing_elements.append(0)
        return list_of_missing_elements

    def test_data(self): #dwie dlugosci
        start = np.random.randint(0, self.missing_scope)
        end = np.random.randint(self.length_of_signal * 2 - self.missing_scope, self.length_of_signal * 2)

        count_of_gaps = self.length_of_signal * 2 // self.missing_scope
        count_of_remove_scopes = int(count_of_gaps * (self.missing_percent / 10))
        length_of_gaps = (end - start - count_of_remove_scopes * self.missing_scope) // (count_of_remove_scopes - 1)
        elements_to_remove = self.list_to_remove(end, length_of_gaps, count_of_remove_scopes)
        return self.list_of_blank_gaps(elements_to_remove)

    def train_data(self, music1, music2):
        X0, X1, y0, y1 = self.create_train_dataset(pd.concat([music1.iloc[0: self.length_of_signal, :],
                                                              music2.iloc[0:self.length_of_signal, :]], axis=0))
        return (X0 + X1) // 2, (y0 + y1) // 2







