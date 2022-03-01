import numpy as np
from keras.saving.save import load_model
from models import coeff_determination
from tqdm import tqdm
from namesOfFiles import name_missing_percent


def prediction(model, name_of_model, original_signal_clone, train_data_set_up, i):
    if name_of_model == "ann":
        return int(model.predict(np.reshape(original_signal_clone[i], (1, train_data_set_up.look_back))))
    else:
        return int(model.predict(np.reshape(original_signal_clone[i], (-1, 1, train_data_set_up.look_back))))


def insertion(train_data_set_up, original_signal_clone, blank_gaps, name_of_model):
    loss = []
    model = load_model('models/' + name_of_model + '_' + name_missing_percent(train_data_set_up) + '.h5',
                       custom_objects={"coeff_determination": coeff_determination})
    for i in tqdm(range(train_data_set_up.length_of_signal * 2 - train_data_set_up.look_back * 5)):
        if blank_gaps[i + 1] == 1:
            pred = prediction(model, name_of_model, original_signal_clone, train_data_set_up, i)
            loss.append(abs(original_signal_clone[i + 1][train_data_set_up.look_back - 1] - pred))
            for lb in range(train_data_set_up.look_back):
                original_signal_clone[i + 1 + lb][train_data_set_up.look_back - 1 - lb] = pred
    return original_signal_clone, loss
