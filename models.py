import os
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU
from tensorflow.keras.models import Sequential
from scores import display, coeff_determination
from namesOfFiles import name_missing_percent_model, name_of_folder


def rnn_R(X, y, epochs, train_data_set_up):
    rnnR = Sequential()
    rnnR.add(LSTM(units=100, activation='relu', input_shape=(None, train_data_set_up.look_back)))
    rnnR.add(Dense(units=50, activation='relu'))
    rnnR.add(Dense(units=25, activation='relu'))
    rnnR.add(Dense(units=12, activation='relu'))
    rnnR.add(Dense(units=1, activation='relu'))

    rnnR.compile(optimizer='adam', loss='mean_squared_error',
                 metrics=['mean_squared_error', 'mean_absolute_error', coeff_determination])
    history = rnnR.fit(X, y, epochs=epochs, batch_size=100)
    display(history, "_rnnR", train_data_set_up)
    rnnR.save('models/' + name_of_folder(train_data_set_up) + '/rnnR_'
              + name_missing_percent_model(train_data_set_up) + '.h5')


def rnn_L(X, y, epochs, train_data_set_up):
    rnnL = Sequential()
    rnnL.add(LSTM(units=100, activation='linear', input_shape=(None, train_data_set_up.look_back)))
    rnnL.add(LeakyReLU())
    rnnL.add(Dense(units=50, activation='linear'))
    rnnL.add(LeakyReLU())
    rnnL.add(Dense(units=25, activation='linear'))
    rnnL.add(LeakyReLU())
    rnnL.add(Dense(units=12, activation='linear'))
    rnnL.add(LeakyReLU())
    rnnL.add(Dense(units=1, activation='linear'))
    rnnL.add(LeakyReLU())

    rnnL.compile(optimizer='adam', loss='mean_squared_error',
                 metrics=['mean_squared_error', 'mean_absolute_error', coeff_determination])
    history = rnnL.fit(X, y, epochs=epochs, batch_size=100)
    display(history, "_rnnL", train_data_set_up)
    rnnL.save('models/' + name_of_folder(train_data_set_up) + '/rnnL_'
              + name_missing_percent_model(train_data_set_up) + '.h5')


def ann_M(X, y, epochs, train_data_set_up):
    ann = Sequential()
    ann.add(Dense(units=100, activation='linear', input_dim=train_data_set_up.look_back))
    ann.add(LeakyReLU())
    ann.add(Dense(units=50, activation='linear'))
    ann.add(LeakyReLU())
    ann.add(Dense(units=25, activation='linear'))
    ann.add(LeakyReLU())
    ann.add(Dense(units=12, activation='linear'))
    ann.add(LeakyReLU())
    ann.add(Dense(units=1, activation='linear'))
    ann.add(LeakyReLU())

    ann.compile(optimizer='adam', loss='mean_squared_error',
                metrics=['mean_squared_error', 'mean_absolute_error', coeff_determination])
    history = ann.fit(X, y, epochs=epochs, batch_size=100)
    display(history, "_ann", train_data_set_up)
    ann.save('models/' + name_of_folder(train_data_set_up) + '/ann_'
             + name_missing_percent_model(train_data_set_up) + '.h5')


def training(ann, rnnR, rnnL, X, y, epochs, train_data_set_up):
    if ann:
        if not os.path.isfile('./models/' + name_of_folder(train_data_set_up) + '/ann_'
                              + name_missing_percent_model(train_data_set_up) + '.h5'):
            ann_M(X, y, epochs, train_data_set_up)
    if rnnR:
        if not os.path.isfile('./models/' + name_of_folder(train_data_set_up) + '/rnnR_'
                              + name_missing_percent_model(train_data_set_up) + '.h5'):
            rnn_R(X.reshape((-1, 1, train_data_set_up.look_back)), y, epochs, train_data_set_up)
    if rnnL:
        if not os.path.isfile('./models/' + name_of_folder(train_data_set_up) + '/rnnL_'
                              + name_missing_percent_model(train_data_set_up) + '.h5'):
            rnn_L(X.reshape((-1, 1, train_data_set_up.look_back)), y, epochs, train_data_set_up)


