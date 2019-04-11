# import pandas as pd  # pandas does things with matrixes
import numpy as np  # used for sorting a matrix
import matplotlib.pyplot as plt  # matplotlib is used for plotting data
# import matplotlib.ticker as ticker  # used for changing tick spacing
import datetime as dt  # used for dates
import matplotlib.dates as mdates  # used for dates, in a different way
import warnings
warnings.filterwarnings("ignore")


def simple_plot(dataframe, name, feature, interval_day=60):
    # makes matrix with only the stock info
    allstocksingle = dataframe[dataframe['Name'] == name]

    x = allstocksingle['Date']
    y = allstocksingle[feature]  # splots which ever catagory you entered above

    plt.figure(figsize=(12, 6))

    # display the date properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    # x axis tick every 60 days
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval_day))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(100))
    # sets y axis tick spacing to 100

    # y_max_lim = dataframe[dataframe['Name'] == name][feature].max() + \
    #     dataframe[dataframe['Name'] == name][feature].max()/10
    # y_min_lim = dataframe[dataframe['Name'] == name][feature].min() - \
    #     dataframe[dataframe['Name'] == name][feature].min()/10

    plt.plot(x, y)  # plots the x and y
    plt.grid(True)  # turns on axis grid
    # rotates the x axis ticks 90 degress and font size 10
    plt.xticks(rotation=90, fontsize=10)
    plt.title(name + ' Stock')  # prints the title on the top
    plt.ylabel(feature + ' Price')  # labels y axis
    plt.xlabel('Date')  # labels x axis


def simple_plot_by_date(dataframe, name, feature, start_day, end_day,
                        interval_day=60):
    # enter the start date here, it must be YYYY-MM-DD
    startdate = dt.datetime.strptime(start_day, '%Y-%m-%d').date()
    # enter the end date here, it must be YYYY-MM-DD
    enddate = dt.datetime.strptime(end_day, '%Y-%m-%d').date()

    # makes matrix with only the stock info
    allstocksingle = dataframe[dataframe['Name'] == name]

    x = allstocksingle['Date']
    # plots which ever catagory you entered above
    y = allstocksingle[feature]

    plt.figure(figsize=(12, 6))

    # display the date properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    # x axis tick every 60 days
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval_day))

    y_max_lim = dataframe[dataframe['Name'] == name][feature].max() + \
        dataframe[dataframe['Name'] == name][feature].max()/10
    y_min_lim = dataframe[dataframe['Name'] == name][feature].min() - \
        dataframe[dataframe['Name'] == name][feature].min()/10

    plt.ylim(y_min_lim, y_max_lim)
    plt.xlim(startdate, enddate)
    plt.plot(x, y)  # plots the x and y
    plt.grid(True)  # turns on axis grid
    # rotates the x axis ticks 90 degress and font size 10
    plt.xticks(rotation=90, fontsize=10)
    plt.title(name + ' Stock')  # prints the title on the top
    plt.ylabel(feature + ' Price')  # labels y axis
    plt.xlabel('Date')  # labels x axis


# Defining a hit counter function
def hit_count(predictions, real):
    number_of_hits = 0
    up_hit = down_hit = 0
    for i in range(1, len(predictions)):
        up_hit = predictions[i-1] > predictions[i] and real[i-1] > real[i]
        down_hit = predictions[i-1] < predictions[i] and real[i-1] < real[i]
        if up_hit or down_hit:
            number_of_hits += 1
    return number_of_hits


def TU(z, z_hat):
    if isinstance(z, list):
        z = np.array(z)
    if isinstance(z_hat, list):
        z_hat = np.array(z_hat)
    num = np.sum((z - z_hat) ** 2) + 1e-6
    den = np.sum((z[1:] - z[:(len(z) - 1)]) ** 2) + 1e-6

    return num/den
