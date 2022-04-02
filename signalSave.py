import numpy as np
import copy
from dataToFromFiles import save_data
from dataToFromFiles import data_to_npy
from predict import insertion
from namesOfFiles import name_missing_percent, name_length_of_signal


def blank_wave(train_data_set_up, blank_signal, blank_gaps):
    for i in range(train_data_set_up.length_of_signal * 2 - train_data_set_up.look_back * 3):
        if blank_gaps[i + 1] == 1:
            for lb in range(train_data_set_up.look_back):
                blank_signal[i+1+lb][train_data_set_up.look_back - 1 - lb] = 0
    return blank_signal


def blank_signal_save(train_data_set_up, original_signal, blank_gaps):
    blank_signal = blank_wave(train_data_set_up, copy.deepcopy(original_signal), blank_gaps)
    save_data(blank_signal[:, train_data_set_up.look_back - 1],
              name_missing_percent(train_data_set_up) + "_blank_gaps")
    data_to_npy(blank_signal, name_missing_percent(train_data_set_up) + "_blank_gaps")


def filled_signal_save(train_data_set_up, original_signal, blank_gaps, name_of_model):
    filled_signal, loss = insertion(train_data_set_up, copy.deepcopy(original_signal), blank_gaps, name_of_model)
    data_to_npy(filled_signal, name_missing_percent(train_data_set_up) + "_" + name_of_model + "_filled_signal")
    data_to_npy(np.array(loss), name_missing_percent(train_data_set_up) + "_" + name_of_model + "_loss")
    save_data(filled_signal[:, train_data_set_up.look_back - 1],
              name_missing_percent(train_data_set_up) + '_' + name_of_model)


def original_signal_save(train_data_set_up, original_signal):
    save_data(original_signal, "original_" + name_length_of_signal(train_data_set_up))


def predict_and_save_signals(ann, rnnR, rnnL, original_signal, blank_gaps, train_data_set_up):
    if ann:
        filled_signal_save(train_data_set_up, original_signal, blank_gaps, "ann")
    if rnnR:
        filled_signal_save(train_data_set_up, original_signal, blank_gaps, "rnnR")
    if rnnL:
        filled_signal_save(train_data_set_up, original_signal, blank_gaps, "rnnL")
