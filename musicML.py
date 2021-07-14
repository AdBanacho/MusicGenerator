from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def metric(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse =  mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return round(mse, 3), round(mae, 3), round(r2, 3)


def knn(X, y, test):
    return neighbors.KNeighborsRegressor(5).fit(X, y).predict(test)


def lin(X, y, test):
    return LinearRegression().fit(X, y).predict(test)


def forest(X, y, test):
    return RandomForestRegressor().fit(X, y).predict(test)


def metrics_to_fail(y, pred_knn, pred_lin, pred_rf, procent):
    pred_knn = pred_knn[:y.shape[0]]
    pred_lin = pred_lin[:y.shape[0]]
    pred_rf = pred_rf[:y.shape[0]]
    with open("scores/scores_ml_" + str(procent) + ".txt", "w") as output:
        output.write("metriki przeprowadzone na " + str(y[:pred_knn.shape[0]].shape[0]) + " danych\n")
        output.write('mean_squared_error, mean_absolute_error, R^2\n')
        output.write('pred_knn ' + str(metric(y, pred_knn)) + '\n')
        output.write('pred_lin ' + str(metric(y, pred_lin)) + '\n')
        output.write('pred_rf ' + str(metric(y, pred_rf)) + '\n')


def ml(X, y, test, procent):
    print("knn")
    pred_knn = knn(X, y, test)
    print("lin")
    pred_lin = lin(X, y, test)
    print("forest")
    pred_rf = forest(X, y, test)
    # print("metriki przeprowadzone na " + str(y[:pred_knn.shape[0]].shape[0]) + " danych")
    # metrics_to_fail(y, pred_knn, pred_lin, pred_rf, procent)
    return pred_knn, pred_lin, pred_rf