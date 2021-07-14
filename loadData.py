import numpy as np
import random
import os
import pandas as pd
from tqdm import tqdm
from scipy.io.wavfile import write, read


def list_to_remove(end, blok, odstepy_dlugosc, ilosc_usuniec):
    list = []
    for i in range(ilosc_usuniec):
        g1 = end - odstepy_dlugosc * i - blok * (i + 1)
        g2 = end - odstepy_dlugosc * i - blok * i
        for j in range(g1, g2):
            list.append(j)
    return list


def losuj_kolejnosc(ile, data):
    return random.sample(range(data), ile)


def create_train_dataset(df, look_back):
    dataX1, dataX2 , dataY1 , dataY2 = [],[],[],[]
    for i in tqdm(range(len(df)-look_back-1)):
        dataX1.append(df.iloc[i : i + look_back, 0].values)
        dataX2.append(df.iloc[i : i + look_back, 1].values)
        dataY1.append(df.iloc[i + look_back, 0])
        dataY2.append(df.iloc[i + look_back, 1])
    return np.array(dataX1), np.array(dataX2), np.array(dataY1), np.array(dataY2)


def create_test_data(df, look_back):
    dataX1, dataX2 = [], []
    for i in tqdm(range(len(df) - look_back - 1)):
        dataX1.append(df.iloc[i: i + look_back, 0].values)
        dataX2.append(df.iloc[i: i + look_back, 1].values)

    return np.array(dataX1), np.array(dataX2)


def load_from_wav(dlugosc):
    rate, music1 = read('sounds/Numb.wav')
    rate, music2 = read('sounds/Eminem.wav')
    music1 = pd.DataFrame(music1[0:dlugosc, :])
    music2 = pd.DataFrame(music2[0:dlugosc, :])
    return music1, music2, rate


def load_data(procent_brakujacy, dlugosc):
    music1, music2, rate = load_from_wav(dlugosc)
    save_orginal(music1, music2, dlugosc)
    ile = int(procent_brakujacy * music1.shape[0])
    usun = losuj_kolejnosc(ile, music1.shape[0])
    music1 = music1.drop(usun)
    music2 = music2.drop(usun)

    X1, X2, y1, y2 = create_train_dataset(pd.concat([music1.iloc[0: dlugosc, :], music2.iloc[0: dlugosc, :]], axis=0),
                                          3)
    return (X1 + X2) // 2, (y1 + y2) // 2, rate


def load_data_cyklicznie(procent_brakujacy, blok, dlugosc):
    music1, music2, rate = load_from_wav(dlugosc)
    save_orginal(music1, music2, dlugosc)

    start = np.random.randint(0, blok)
    end = np.random.randint(dlugosc - blok, dlugosc)

    ilosc_fragmentow = dlugosc // blok
    ilosc_usuniec = int(ilosc_fragmentow * procent_brakujacy)
    odstepy_dlugosc = (end - start - ilosc_usuniec * blok) // (ilosc_usuniec - 1)
    usun = list_to_remove(end, blok, odstepy_dlugosc, ilosc_usuniec)
    print(music1.shape)
    music1 = music1.drop(usun)
    music2 = music2.drop(usun)
    print(music1.shape)
    X1, X2, y1, y2 = create_train_dataset(pd.concat([music1.iloc[0: dlugosc, :], music2.iloc[0: dlugosc, :]], axis=0),
                                          3)
    return (X1 + X2) // 2, (y1 + y2) // 2, rate


def save_data(rate, pred, name):
    write('predict_sounds/' + name + '.wav', rate, pd.concat([pd.DataFrame(pred.astype('int16'))], axis=1).values)


def save_orginal(music1, music2, dlugosc):
    if not os.path.isfile('./npy/original' + str(dlugosc) + '.npy'):
        Xo, Xo, yo, yo = create_train_dataset(pd.concat([music1.iloc[0: dlugosc, :], music2.iloc[0: dlugosc, :]], axis=0),
                                          3)
        original = (Xo + Xo) // 2
        data_to_npy(original, "original" + str(dlugosc))


def data_to_npy(data, name):
    np.save('npy/' + name + '.npy', data)


def data_from_npy(name):
    return np.load('npy/' + name + '.npy')


def data_from_wav_to_npy(procent_brakujacy, dlugosc, blok):
    if blok == 0:
        X, y, rate = load_data(procent_brakujacy / 10, dlugosc)
    else:
        X, y, rate = load_data_cyklicznie(procent_brakujacy / 10, blok, dlugosc)
    data_to_npy(X, "X" + str(procent_brakujacy))
    data_to_npy(y, "y" + str(procent_brakujacy))
    data_to_npy(rate, "rate" + str(procent_brakujacy))


def data_set_from_npy(procent_brakujacy, dlugosc):
    X = data_from_npy("X" + str(procent_brakujacy))
    y = data_from_npy("y" + str(procent_brakujacy))
    rate = data_from_npy("rate" + str(procent_brakujacy))
    original = data_from_npy("original" + str(dlugosc))
    return X, y, rate, original
