import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


#Importing the dataSet
def load_data():
    dataset_flight_info = pd.read_csv('Data_test_Analytics.csv')
    dataset_route_info = pd.read_csv('RouteInfo_test_Analytics.csv')

    #merge the flight info and he route info
    dataset_x = pd.merge(dataset_flight_info, dataset_route_info, on='Route')


    #Handling categorical date
    dataset_onehot = dataset_x.copy()
    dataset_onehot = pd.get_dummies(dataset_onehot, columns=['Country', 'Region', 'Route'], prefix=['Country', 'Region', 'Route'])

    # DepartureDate
    dataset_date = dataset_onehot['DepartureDate'].str.split('/',  expand=True)

    dataset = pd.concat([dataset_date, dataset_onehot.iloc[:, 1:]], axis=1, sort=False)

    #drop rows with NA value(only 2 present)
    dataset = dataset.dropna(axis=0)
    print(dataset.info())
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 6].values

    return x, y


def pre_processing(y):
    imputer = Imputer(missing_values='NA', strategy='mean', axis=0)
    imputer = imputer.fit(y[:, 1])
    y[:, 1] = imputer.transform(y[:, 1])
    return y

def fit_model(x_tran, y_tran):
    # Fitting Decision Tree Regression to the dataset
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(x_tran, y_tran)

    return regressor


x, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

[y_rows] = y_test.shape


regressor = fit_model(X_train, y_train)

y_train_cal = regressor.predict(X_train)
rms = sqrt(mean_squared_error(y_train, y_train_cal))
print(rms)

# Predicting a new result
y_pred = regressor.predict(X_test)

rms = sqrt(mean_squared_error(y_test, y_pred))

print(rms)
