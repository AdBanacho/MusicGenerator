from tensorflow.keras.layers import Dense, LSTM, LeakyReLU
from tensorflow.keras.models import Sequential
from scores import coeff_determination, display


def rnn_R(X, y, test, epochs, procent_brakujacy):
    rnnR = Sequential()
    rnnR.add(LSTM(units=100, activation='relu', input_shape=(None, 3)))
    rnnR.add(Dense(units=50, activation='relu'))
    rnnR.add(Dense(units=25, activation='relu'))
    rnnR.add(Dense(units=12, activation='relu'))
    rnnR.add(Dense(units=1, activation='relu'))
    rnnR.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mean_absolute_error',
                                                                            coeff_determination])
    history = rnnR.fit(X, y, epochs=epochs, batch_size=100)
    display(history, 'rnnR' + str(procent_brakujacy))

    return rnnR.predict(test.reshape(-1, 1, 3))


def rnn_L(X, y, test, epochs, procent_brakujacy):
    rnnL = Sequential()
    rnnL.add(LSTM(units=100, activation='linear', input_shape=(None, 3)))
    rnnL.add(LeakyReLU())
    rnnL.add(Dense(units=50, activation='linear'))
    rnnL.add(LeakyReLU())
    rnnL.add(Dense(units=25, activation='linear'))
    rnnL.add(LeakyReLU())
    rnnL.add(Dense(units=12, activation='linear'))
    rnnL.add(LeakyReLU())
    rnnL.add(Dense(units=1, activation='linear'))
    rnnL.add(LeakyReLU())

    rnnL.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mean_absolute_error',
                                                                       coeff_determination])
    history = rnnL.fit(X, y, epochs=epochs, batch_size=100)
    display(history, 'rnnL' + str(procent_brakujacy))
    return rnnL.predict(test.reshape(-1, 1, 3))


def ann(X, y, test, epochs, procent_brakujacy):
    ann = Sequential()
    ann.add(Dense(units=100, activation='linear', input_dim=3))
    ann.add(LeakyReLU())
    ann.add(Dense(units=50, activation='linear'))
    ann.add(LeakyReLU())
    ann.add(Dense(units=25, activation='linear'))
    ann.add(LeakyReLU())
    ann.add(Dense(units=12, activation='linear'))
    ann.add(LeakyReLU())
    ann.add(Dense(units=1, activation='linear'))
    ann.add(LeakyReLU())

    ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mean_absolute_error',
                                                                       coeff_determination])
    history = ann.fit(X, y, epochs=epochs, batch_size=100)
    display(history, 'ann' + str(procent_brakujacy))
    return ann.predict(test)



