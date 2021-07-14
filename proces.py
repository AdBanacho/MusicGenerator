from loadData import data_set_from_npy, save_data, data_from_wav_to_npy
from musicDL import rnn_R, rnn_L, ann
#from musicML import ml

def run(epoki, dlugosc, blok, DL, ML, create_data):
    for procent_brakujacy in range(1, 10):

        # utworzenie danych
        if create_data:
            data_from_wav_to_npy(procent_brakujacy, dlugosc, blok)

        # wczytanie danych
        X, y, rate, test = data_set_from_npy(procent_brakujacy, dlugosc)
        save_data(rate, y, "yyyyyyy0" + str(procent_brakujacy))
        if DL:
            pred_rnnR = rnn_R(X.reshape((-1, 1, 3)), y, test.reshape((-1, 1, 3)), epoki, procent_brakujacy)
            pred_rnnL = rnn_L(X.reshape((-1, 1, 3)), y, test.reshape((-1, 1, 3)), epoki, procent_brakujacy)
            pred_ann = ann(X.reshape((-1, 1, 3)), y, test, epoki, procent_brakujacy)

            save_data(rate, pred_rnnR, "rnnR0" + str(procent_brakujacy))
            save_data(rate, pred_rnnL, "rnnL0" + str(procent_brakujacy))
            save_data(rate, pred_ann, "ann0" + str(procent_brakujacy))

        # if ML: # czeka na zmiane biblioteki
        #     pred_knn, pred_lin, pred_rf = ml(X, y, test, procent_brakujacy)
        #
        #     save_data(rate, pred_knn, "knn0" + str(procent_brakujacy))
        #     save_data(rate, pred_lin, "lin0" + str(procent_brakujacy))
        #     save_data(rate, pred_rf, "rf0" + str(procent_brakujacy))
        #     # musi wejsc tyle samo danych tesowych co predykcyjnych y.shape = test.shape, tylko ze y shape to sa dzwieki z
        #     # losowo brakujacymi elementami sciezki orginalnej (test)

        print("-----------------> " + str(procent_brakujacy) + "0% <-------------------")


