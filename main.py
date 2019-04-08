import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import datetime as d


def load_dataset():
    dataset_flight_info = pd.read_csv('Data_test_Analytics.csv')
    dataset_route_info = pd.read_csv('RouteInfo_test_Analytics.csv')

    dataset = pd.merge(dataset_flight_info, dataset_route_info, on='Route')
    return dataset


def pre_processing(dataset, result_dimension):
    if result_dimension == '3':
        region_to_filter_option = input('Enter Region Name\n1.Europe\n2.South\n')
        if region_to_filter_option == '1':
            region_to_filter = 'Europe'
        elif region_to_filter_option == '2':
            region_to_filter = 'South'

        dataset = dataset[dataset['Region'] == region_to_filter]

    elif result_dimension == '4':
        region_to_filter_option = input('Enter Country Name\n1.France\n2.UK\n3.Mexico\n')
        if region_to_filter_option == '1':
            country_to_filter = 'France'
        if region_to_filter_option == '2':
            country_to_filter = 'UK'
        if region_to_filter_option == '3':
            country_to_filter = 'Mexico'
        dataset = dataset[dataset['Country'] == country_to_filter]

    cols_to_drop = ['FlightNumber', 'Route', 'Hour', 'Capacity', 'Country', 'Region']
    dataset.drop(cols_to_drop, axis=1, inplace=True)

    dataset["DepartureDate"] = pd.to_datetime(dataset["DepartureDate"])
    dataset = dataset.dropna(axis=0)
    dataset = dataset.groupby('DepartureDate')['Booked'].sum().reset_index()
    dataset = dataset.set_index('DepartureDate')
    dataset = dataset['Booked'].resample('W').sum()
    dataset = dataset.dropna(axis=0)
    dataset = dataset[dataset > 0]
    return dataset


def display_result(result_dimension, dataset_val2, dataset_test, predictions_val):
    plt.figure(figsize=(16, 8))
    if result_dimension == '1':
        dataset_val2 = dataset_val2.resample('M').sum()
        dataset_test = dataset_test.resample('M').sum()
        predictions_val = predictions_val.resample('M').sum()
        ax = dataset_val2.plot(label='2016')
        ax = dataset_test.plot(label='2017')
        predictions_val.plot(ax=ax, label='Predicted Test', alpha=.7, figsize=(14, 7))
    elif result_dimension == '2':
        dataset_test = pd.concat([dataset_test, predictions_val, dataset_val2])
        dataset_test = dataset_test.resample('Y').sum()
        ax = dataset_test.plot(label='Booking')
    else:
        ax = dataset_val2.plot(label='2016')
        ax = dataset_test.plot(label='2017')
        predictions_val.plot(ax=ax, label='Predicted Test', alpha=.7, figsize=(14, 7))

    plt.show()


result_dimension = input("forecast results dimensions?Enter:\n1.Monthly\n2. Yearly\n3. Region\n4.Country\n")
dataset = load_dataset()
dataset = pre_processing(dataset, result_dimension)

dataset_val2 = dataset[0:28]
dataset_test = dataset[28:55]
dataset_val = dataset

# m =28 for weeks and 184 for days
mod = auto_arima(dataset_val,
                 start_p=1,
                 max_p=8,
                 d=0,
                 max_d=3,
                 start_q=0,
                 max_q=3,
                 seasonal=True,
                 m=28,
                 start_P=0,
                 D=1,
                 trace=True,
                 error_action='ignore',
                 suppress_warnings=True,
                 stepwise=True,
                 out_of_sample_size=10)

print('AIC is ' + str(mod.aic()))

forcast = mod.predict(n_periods=27)

base = pd.to_datetime('2018 05 01')
date_list = [base + d.timedelta(weeks=x) for x in range(0, 27)]
predictions_val = pd.DataFrame(forcast)
predictions_val['DepartureDate'] = date_list
predictions_val = predictions_val.set_index('DepartureDate')

display_result(result_dimension, dataset_val2, dataset_test, predictions_val)

print(dataset_val2.describe())
print(dataset_test.describe())
print(predictions_val.describe())

print('sum in 2016 is ' + str(dataset_val2.sum()))
print('sum in 2017 is ' + str(dataset_test.sum()))
print('sum in 2018 is ' + str(predictions_val.sum()))
