import os
import numpy as np
import pandas as pd
from scipy.io.wavfile import write, read
from namesOfFiles import name_missing_percent, name_length_of_signal


def load_from_wav(length_of_signal):
    rate, music1 = read('sounds/Numb.wav')
    rate, music2 = read('sounds/Eminem.wav')
    music1 = pd.DataFrame(music1[200000:length_of_signal + 200000, :])
    music2 = pd.DataFrame(music2[200000:length_of_signal + 200000, :])
    return music1, music2, rate


def save_data(pred, name):
    write('predict_sounds/' + name + '.wav',
          data_from_npy("rate"),
          pd.concat([pd.DataFrame(pred.astype('int16'))], axis=1).values)


def data_to_npy(data, name):
    np.save('npy/' + name + '.npy', data)


def data_from_npy(name):
    return np.load('npy/' + name + '.npy')


def create_data_from_wav_to_npy(train_data_set_up):
    music1, music2, rate = load_from_wav(train_data_set_up.length_of_signal)
    blank_gaps = train_data_set_up.test_data()
    if not os.path.isfile('./npy/originalX_' + name_length_of_signal(train_data_set_up) + '.npy'):
        X, y = train_data_set_up.train_data(music1, music2)
        data_to_npy(X, "originalX_" + name_length_of_signal(train_data_set_up))
        data_to_npy(y, "originalY_" + name_length_of_signal(train_data_set_up))
    data_to_npy(blank_gaps, "blank_gaps_" + name_missing_percent(train_data_set_up))
    data_to_npy(rate, "rate")


def load_data_from_npy(train_data_set_up):
    X = data_from_npy("originalX_" + name_length_of_signal(train_data_set_up))
    y = data_from_npy("originalY_" + name_length_of_signal(train_data_set_up))
    blank_gaps = data_from_npy("blank_gaps_" + name_missing_percent(train_data_set_up))
    return X, y, blank_gaps

