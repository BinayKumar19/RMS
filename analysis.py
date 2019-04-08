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
        region_to_filter_option = input('Enter Region Name/n1.Europe\n2.South\n')
        if region_to_filter_option == '1':
            region_to_filter = 'Europe'
        elif region_to_filter_option == '2':
            region_to_filter = 'South'

        dataset = dataset[dataset['Region'] == region_to_filter]

    elif result_dimension == '4':
        region_to_filter_option = input('Enter Country Name/n1.France\n2.UK\n3.Mexico\n')
        if region_to_filter_option == '1':
            country_to_filter = 'France'
        if region_to_filter_option == '2':
            country_to_filter = 'UK'
        if region_to_filter_option == '3':
            country_to_filter = 'Mexico'
        if region_to_filter_option == '4':
            country_to_filter = 'France'
        dataset = dataset[dataset['Country'] ==country_to_filter]

    dataset['Difference'] = dataset['Capacity'] - dataset['Booked']

#    cols_to_drop = ['FlightNumber', 'Route', 'Hour', 'Country', 'Region', 'Capacity', 'Booked']
    cols_to_drop = ['FlightNumber', 'Route', 'Hour', 'Country', 'Region']
    dataset.drop(cols_to_drop, axis=1, inplace=True)

    dataset["DepartureDate"] = pd.to_datetime(dataset["DepartureDate"])
    dataset = dataset.dropna(axis=0)
    dataset = dataset.sort_values('DepartureDate')
    #dataset.isnull().sum()
    #dataset = dataset.dropna(axis=0)
    #dataset = dataset[dataset > 0]
    return dataset

def display_result(dataset):

    plt.figure(figsize=(16, 8))
    ax = dataset.plot(label='Observed')
    plt.show()

result_dimension = input("forecast results dimensions?Enter:\n1.Monthly\n2. Yearly\n3. Region\n4.Country\n")
dataset = load_dataset()
dataset = pre_processing(dataset, result_dimension)
dataset = dataset.set_index('DepartureDate')
if result_dimension == '1':
    dataset = dataset.resample('M').sum()
    dataset = dataset[dataset != 0]
    dataset_val2 = dataset[0:6]
    dataset_test = dataset[6:]
elif result_dimension == '2':
    dataset = dataset.resample('Y').sum()
    dataset_val2 = dataset[0:1]
    dataset_test = dataset[1:]
else:
    dataset = dataset.resample('Y').sum()
    dataset_val2 = dataset[0:1]
    dataset_test = dataset[1:]

display_result(dataset)

print(dataset_val2.describe())
print(dataset_test.describe())

print('sum in 2016 is ' + str(dataset_val2.sum()))
print('sum in 2017 is ' + str(dataset_test.sum()))
