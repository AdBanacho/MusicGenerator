import keras.backend as K
import matplotlib.pyplot as plt
from namesOfFiles import name_missing_percent


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def display(history, name, train_data_setup):

    plt.subplot(2, 2, 1)
    plt.plot(history.history['mean_squared_error'])
    plt.title('mean squared error')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 2)
    plt.plot(history.history['mean_absolute_error'])
    plt.title('mean absolute error')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 3)
    plt.plot(history.history['coeff_determination'])
    plt.title('coeff determination')
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.savefig('scores/' + name_missing_percent(train_data_setup) + name + '.png')
    plt.close()

    with open('scores/' + name_missing_percent(train_data_setup) + name + '.txt', "w") as output:
        output.write('Predict ' + name_missing_percent(train_data_setup) + name + '\n')
        output.write('Mean squared error: ' + str(round(history.history['loss'][-1], 3)) + '\n')
        output.write('Mean absolute error: ' + str(round(history.history['mean_absolute_error'][-1], 3)) + '\n')
        output.write('R^2: ' + str(round(history.history['coeff_determination'][-1], 3)) + '\n')
