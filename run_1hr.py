from statistics import variance
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import read_excel
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

def timeseries_to_supervised(data, lag=1):
	ret = np.zeros((data.shape[0]-4, 2))
	for i in range(ret.shape[0]):
		ret[i, 0] = data[i]
		ret[i, 1] = data[i+4]
	return ret

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)



# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]



# scale train and test data to [-1, 1]
def scale(test):
    scaler = pickle.load(open("scaler.pickle", 'rb'))
    test_scaled = scaler.transform(test)
    return scaler, test_scaled



# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]



# retreiving saved model
def retrieve_model():
    model = pickle.load(open('trained_model.pickle', "rb"))
    return model



# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


def uploadfn():
	return filedialog.askopenfilename()


def browse_directory():
	return filedialog.askdirectory()


file_loc = uploadfn()

#read_data
series = read_excel(file_loc)

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)

# retrieving_test_set
test = supervised

# transform the scale of the data
scaler, test_scaled = scale(test)


# retrieving the model from saved file
lstm_model = retrieve_model()


# checking result on test set to test accuracy
pred = np.zeros((len(test_scaled), 1))
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    pred[i][0] = yhat

save_loc = browse_directory()
df = DataFrame(pred)
df.to_excel(save_loc+"predictions_1hr.xlsx", index=False, sheet_name="Results")
