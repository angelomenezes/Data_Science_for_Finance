def simple_plot(dataframe, name, feature, interval_day = 60):
    
    allstocksingle = dataframe[dataframe['Name'] == name] #makes matrix with only the stock info
    
    
    #x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in allstocksingle['Date']] #convert date to something python understands
    x = allstocksingle['Date']
    y = allstocksingle[feature] #plots which ever catagory you entered above

    plt.figure(figsize=(12,6))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y')) #display the date properly
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = interval_day)) #x axis tick every 60 days
    #plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(100)) # sets y axis tick spacing to 100
    
    y_max_lim = dataframe[dataframe['Name'] == name][feature].max() + dataframe[dataframe['Name'] == name][feature].max()/10
    y_min_lim = dataframe[dataframe['Name'] == name][feature].min() - dataframe[dataframe['Name'] == name][feature].min()/10
    
    #plt.ylim(y_min_lim,y_max_lim)
    #plt.xlim(startdate, enddate)
    plt.plot(x,y) #plots the x and y
    plt.grid(True) #turns on axis grid
    plt.xticks(rotation=90,fontsize = 10) #rotates the x axis ticks 90 degress and font size 10
    plt.title(name + ' Stock') #prints the title on the top
    plt.ylabel(feature + ' Price') #labels y axis
    plt.xlabel('Date') #labels x axis

def simple_plot_by_date(dataframe, name, feature, start_day, end_day, interval_day = 60):
    
    startdate = dt.datetime.strptime(start_day, '%Y-%m-%d').date() #enter the start date here, it must be YYYY-MM-DD
    enddate = dt.datetime.strptime(end_day, '%Y-%m-%d').date() #enter the end date here, it must be YYYY-MM-DD
    
    allstocksingle = dataframe[dataframe['Name'] == name] #makes matrix with only the stock info

    #x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in allstocksingle['Date']] #convert date to something python understands
    x = allstocksingle['Date']
    y = allstocksingle[feature] #plots which ever catagory you entered above

    plt.figure(figsize=(12,6))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y')) #display the date properly
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = interval_day)) #x axis tick every 60 days
    #plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(100)) # sets y axis tick spacing to 100
    
    y_max_lim = dataframe[dataframe['Name'] == name][feature].max() + dataframe[dataframe['Name'] == name][feature].max()/10
    y_min_lim = dataframe[dataframe['Name'] == name][feature].min() - dataframe[dataframe['Name'] == name][feature].min()/10
    
    plt.ylim(y_min_lim,y_max_lim)
    plt.xlim(startdate, enddate)
    plt.plot(x,y) #plots the x and y
    plt.grid(True) #turns on axis grid
    plt.xticks(rotation=90,fontsize = 10) #rotates the x axis ticks 90 degress and font size 10
    plt.title(name + ' Stock') #prints the title on the top
    plt.ylabel(feature + ' Price') #labels y axis
    plt.xlabel('Date') #labels x axis
    

# Defining a hit counter function
def hit_count(predictions, real):    
    number_of_hits = 0
    up_hit = down_hit = 0    
    for i in range(1, len(predictions)):
        up_hit = predictions[i-1]>predictions[i] and real[i-1]>real[i]
        down_hit = predictions[i-1]<predictions[i] and real[i-1]<real[i]
        if up_hit or down_hit:
            number_of_hits +=1
    return number_of_hits