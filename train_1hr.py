from pandas import read_csv
from pandas import DataFrame
from pandas import Series
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import pickle

def to_supervised(data):
    ret = np.zeros((data.shape[0]-4, 2))
    for i in range(ret.shape[0]):
        ret[i, 0] = data[i]
        ret[i, 1] = data[i+4]
    return ret

def scale(train, test):
    scaler = MinMaxScaler(feature_range = (-1,1))
    scaler = scaler.fit(train)
    train = np.reshape(train, (train.shape[0], train.shape[1]))
    test = np.reshape(test, (test.shape[0], test.shape[1]))
    ret_train = scaler.transform(train)
    ret_test = scaler.transform(test)
    pickle.dump(scaler,open('scaler.pickle', 'wb'))
    return scaler, ret_train, ret_test

def inverse_scaling(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def difference(arr, interval= 1):
    ret = np.zeros((arr.shape[0]-interval, arr.shape[1]))
    for i in range(interval, ret.shape[0]):
        ret[i,:] = arr[i+interval, :] - arr[i, :]
    return ret

def reverse_difference(history, val, interval=1):
    return val+history[-interval]

def train_model(train, neurons, batch_len, epochs):
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape = (batch_len, x_train.shape[1], x_train.shape[2]), stateful = True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    for i in range(epochs):
        model.fit(x_train, y_train, epochs=1, batch_size=batch_len, verbose=1, shuffle = False)
        model.reset_states()
    pickle.dump(model, open('model.pickle', 'wb'))
    return model

def predict(model, X, batch_len):
    X = X.reshape(1,1,len(X))
    y_pred = model.predict(X, batch_size = batch_len)
    return y_pred[0,0]


def error_check(test, model, vals, scaler):
    err = list()
    avg_err = 0
    test_x = test[:, :-1]
    y_act = test[:, -1]
    y_predicted = []
    for i in range(3000):
        y_pred = predict(model, test_x[i, :], 1)
        X = test_x[i, :-1]
        y_pred = inverse_scaling(scaler, X, y_pred)
        y_pred = reverse_difference(vals, y_pred, len(test)-i+1)
        y_prediction.append(scaler, y_pred, len(test)+1-i)
        t = abs(y_pred-y_act[i])/y_act[i]
        avg_err += t
        err.append(t)
    avg_err /= 3000
    return err, avg_err, y_prediction

def error_plot(error_list, avg_error, predictions , test_data):
    plt.plot(error_list)
    plt.show()
    plt.plot(predictions)
    plt.plot(test_data[:, -1])
    plt.show()
    print("Average percentage error =", avg_error*100)



def start():
    data = read_csv('time_series.csv')
    raw_data = data.values
    arr = difference(raw_data)
    arr = to_supervised(arr)
    train, test = arr[:-3000, :], arr[-3000:, :]
    scaler, train_scaled, test_scaled = scale(train, test)
    trained_model = train_model(train_scaled, 8, 1, 10)
    #error_plot(error_check(test_scaled, trained_model, raw_data, scaler), test)



if __name__ == '__main__':
    start()
