import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split


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

def fit_model(x_tran, y_tran, x_val):
    # Fitting Decision Tree Regression to the dataset
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(x_tran, y_tran)


    # # Visualising the Decision Tree Regression results (higher resolution)
    # X_grid = np.arange(min(x_tran[:,3]), max(x_tran[:,3]), 0.01)
    # X_grid = X_grid.reshape((len(X_grid), 1))
    # plt.scatter(x_tran[:,3], y_tran, color='red')
    # plt.plot(X_grid, regressor.predict(x_tran), color='blue')
    # plt.title('Truth or Bluff (Decision Tree Regression)')
    # plt.xlabel('Position level')
    # plt.ylabel('Salary')
    # plt.show()

    # Predicting a new result
    y_pred = regressor.predict(x_val)
    return y_pred

x, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x.shape)
print(x[977,:])   #- 1453
print(x[1453,:]) #- 1008

#print(y[977])
#y = pre_processing(y)
y_cal = fit_model(x, y, x)



#print(y_cal)


# x_rows, x_columns = X.shape
# x_tran = np.zeros(shape = (0, 7))
# y_tran = np.zeros(shape = (0, 1))
# x_val = np.zeros(shape = (0, 7))
# y_val = np.zeros(shape = (0, 1))



# for i in range(0, x_rows):
#         datetime_object = datetime.strptime(X[i, 1], '%m/%d/%Y')
#         dateComponents = np.array([datetime_object.year, datetime_object.month, datetime_object.day])
#         if str(datetime_object.year) == '2016':
#             x_tran = np.vstack((x_tran, np.hstack((X[i, 0], dateComponents, X[i, 2:5]))))
#             y_tran = np.vstack((y_tran, y[i]))
#         else:
#             x_val = np.vstack((x_val, np.hstack((X[i, 0], dateComponents, X[i, 2:5]))))
#             y_val = np.vstack((y_val, y[i]))

# print(X.shape)
# print(x_tran.shape)
# print(x_val.shape)
# print(y_tran.shape)
# print(y_val.shape)
#
#
#
#
# # Visualising the Decision Tree Regression results (higher resolution)
# X_grid = np.arange(min(x_tran), max(x_tran), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(x_tran, y_tran, color = 'red')
# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Truth or Bluff (Decision Tree Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

