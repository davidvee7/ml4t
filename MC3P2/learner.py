__author__ = 'davidvinegar'
from sklearn import neighbors

import numpy as np

from sklearn.neighbors import KNeighborsRegressor
import LinRegLearner as lrl
import math
import KNNLearner as knn
import csv as csv
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import scipy.optimize as sco
from util import get_data, plot_data


class learner(object):

    def trade(self,data,unalteredPrices):
        longIsOpen = False
        shortIsOpen = False
        buys = []
        shorts = []
        close = []

        fileName = "./Orders/orders.csv"
        f=open(fileName,"w+")
        writer = csv.writer(f)
        # headerColumns = "Date,Symbol,Order,Shares"
        headerColumns = ["Date","Symbol","Order","Shares"]
        writer.writerow(headerColumns)

        # print "what does data look like"
        # print data

        symbols = ['IBM']

        holdFive = -1
        # print data
        # print unalteredPrices
        unalteredPrices = unalteredPrices[3:-3]
        # print "shapes "
        # print unalteredPrices.shape
        # print data.shape
        # print data.columns
        for i in data.index.date:
            # print i

            #go long
            # print data.ix[i,1]
            if holdFive>=0 and holdFive < 5:
                holdFive +=1
                continue
            else:

                if (data["Predicted Y"].ix[i]> unalteredPrices["IBM"].ix[i] and (longIsOpen==False or shortIsOpen==True)):
                    if shortIsOpen == True:
                        shortIsOpen=False
                        close.append(i)
                        orderDate = str(i)
                        orderType = "BUY"
                        shares = "100"
                        symbol = "IBM"
                        rowValues = [orderDate,symbol,orderType,shares]
                        writer.writerow(rowValues)
                        holdFive=0
                    else:
                        # print "tim eto buy"
                        buys.append(i)
                        longIsOpen = True
                        orderDate = str(i)
                        orderType = "BUY"
                        shares = "100"
                        symbol = "IBM"
                        rowValues = [orderDate,symbol,orderType,shares]
                        writer.writerow(rowValues)
                        holdFive=0

                elif data["Predicted Y"].ix[i]<unalteredPrices["IBM"].ix[i] and (shortIsOpen==False or longIsOpen==True):
                    if longIsOpen == True:
                        close.append(i)
                        # shorts.append(i)
                        orderDate = str(i)
                        orderType = "SELL"
                        shares = "100"
                        symbol = "IBM"
                        longIsOpen=False
                        rowValues = [orderDate,symbol,orderType,shares]
                        writer.writerow(rowValues)
                        holdFive=0

                    else:
                        shorts.append(i)
                        shortIsOpen = True
                        orderDate = str(i)
                        orderType = "SELL"
                        shares = "100"
                        symbol = "IBM"

                        rowValues = [orderDate,symbol,orderType,shares]
                        writer.writerow(rowValues)
                        holdFive=0

        f.close()
        symbols = ['IBM']
        unalteredPrices = get_data(symbols,dates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        unalteredPrices= unalteredPrices/unalteredPrices.ix[0,:]

        ax = unalteredPrices["IBM"].plot(title="Entry/Exit Graph", label = "IBM",color ='b')
        # print buys
        ymin, ymax = ax.get_ylim()
        plt.vlines(buys,ymin=ymin,ymax=ymax, color = 'g')
        plt.vlines(close,ymin=ymin,ymax=ymax, color = 'k')
        legend = ax.legend()

        # print "shorts"
        # print shorts
        plt.vlines(shorts,ymin=ymin,ymax=ymax, color = 'r')
        #
        plt.show()

    def tradeTest(self,data):
        longIsOpen = False
        shortIsOpen = False
        buys = []
        shorts = []
        close = []

        fileName = "./Orders/orderstest.csv"
        f=open(fileName,"w+")
        writer = csv.writer(f)
        # headerColumns = "Date,Symbol,Order,Shares"
        headerColumns = ["Date","Symbol","Order","Shares"]
        writer.writerow(headerColumns)

        testDates=  pd.date_range('2009-12-31', '2011-12-31')

        symbols = ['IBM']
        unalteredPrices = get_data(symbols,testDates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        unalteredPrices= unalteredPrices/unalteredPrices.ix[0,:]

        unalteredPrices = unalteredPrices[3:-3]
        count = 0
        holdFive = -1
        for i in data.index.date:
            # print i

            #go long
            # print data.ix[i,1]
            if holdFive>=0 and holdFive < 5:
                holdFive +=1
                continue
            else:

                if data["Predicted Y"].ix[i]> unalteredPrices["IBM"].ix[i] and longIsOpen==False:
                    if shortIsOpen == True:
                        shortIsOpen=False
                        close.append(i)
                        orderDate = str(i)
                        orderType = "BUY"
                        shares = "100"
                        symbol = "IBM"
                        rowValues = [orderDate,symbol,orderType,shares]
                        writer.writerow(rowValues)
                        holdFive=0
                    else:
                        # print "tim eto buy"
                        buys.append(i)
                        longIsOpen = True
                        orderDate = str(i)
                        orderType = "BUY"
                        shares = "100"
                        symbol = "IBM"
                        rowValues = [orderDate,symbol,orderType,shares]
                        writer.writerow(rowValues)
                        holdFive=0

                elif data["Predicted Y"].ix[i]<unalteredPrices["IBM"].ix[i] and (shortIsOpen==False or longIsOpen==True):
                    if longIsOpen == True:
                        close.append(i)
                        # shorts.append(i)
                        orderDate = str(i)
                        orderType = "SELL"
                        shares = "100"
                        symbol = "IBM"
                        longIsOpen=False
                        rowValues = [orderDate,symbol,orderType,shares]
                        writer.writerow(rowValues)
                        holdFive=0

                    else:
                        shorts.append(i)
                        shortIsOpen = True
                        orderDate = str(i)
                        orderType = "SELL"
                        shares = "100"
                        symbol = "IBM"

                        rowValues = [orderDate,symbol,orderType,shares]
                        writer.writerow(rowValues)
                        holdFive=0

        f.close()
        testDates=  pd.date_range('2009-12-31', '2011-12-31')

        symbols = ['IBM']
        unalteredPrices = get_data(symbols,testDates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        unalteredPrices= unalteredPrices/unalteredPrices.ix[0,:]

        ax = unalteredPrices.plot(title="Entry/Exit Graph", label = "IBM",color ='b')        # print buys
        ymin, ymax = ax.get_ylim()
        plt.vlines(buys,ymin=ymin,ymax=ymax, color = 'g')
        plt.vlines(close,ymin=ymin,ymax=ymax, color = 'k')
        # print "shorts"
        # print shorts
        plt.vlines(shorts,ymin=ymin,ymax=ymax, color = 'r')
        #
        plt.show()

    def setUp(self,dates):
        momentumDF, actual5DayChange, unalteredPrices = self.getMomentum(dates)
        spyMomentum = self.getSPYMomentum(dates)
        stdDF=self.getVolatility(dates)

        SMA= pd.rolling_mean(unalteredPrices, window = 3)
        SMA= SMA.dropna()
        bb_value = (unalteredPrices-SMA)/(2*stdDF)

        stats = []

        bbMean = bb_value.mean()
        bbStd = bb_value.std()
        bb_value = (bb_value-bbMean) / bbStd
        momentumMean = momentumDF.mean()
        momentumStd = momentumDF.std()
        momentumDF = (momentumDF-momentumMean) / momentumStd
        spyMean = spyMomentum.mean()
        spyStd = spyMomentum.std()
        spyDF = (spyMomentum-spyMean)/ spyStd
        volatilityMean = stdDF.mean()
        volatilityStd = stdDF.std()
        stdDF = (stdDF-volatilityMean)/ volatilityStd

        stats.append(momentumMean)
        stats.append(momentumStd)
        stats.append(volatilityMean)
        stats.append(volatilityStd)
        stats.append(bbMean)
        stats.append(bbStd)

        spyDF = spyDF[3:-3]
        momentumDF = momentumDF[3:-3]
        stdDF = stdDF[3:-3]
        actual5DayChange = actual5DayChange[3:-3]
        bb_value = bb_value[3:-3]
        unalteredPrices= unalteredPrices/unalteredPrices.ix[0,:]
        unalteredPrices = unalteredPrices[3:-3]

        actual5DayChange = actual5DayChange+1

        allDF = np.ones((momentumDF.shape[0],4))
        allDF[:,0]= momentumDF['IBM']
        allDF[:,1]= stdDF['IBM']
        allDF[:,2]= bb_value['IBM']
        allDF[:,3]= actual5DayChange['IBM']

        trainX = allDF[:,0:-1]
        trainY = allDF[:,-1]

        # print trainX
        # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
        learner = knn.KNNLearner(2,verbose = True) # create a knn learner
        learner.addEvidence(trainX, trainY) # train it
        predYTrain = learner.query(trainX) # get the predictions        sy = sknn.fit(trainX, trainY).predict(testX)
        # predYTest = learner.query(testX)


        #b3 predytrain is 972, future price df is 471
        yPredDF = pd.DataFrame(predYTrain, index = actual5DayChange.index)

        yPredXPriceDF= yPredDF.values*unalteredPrices
        fiveDayPrices = actual5DayChange.values* unalteredPrices

        yPredXPriceDF.columns = ['Predicted Y']
        fiveDayPrices.columns = ['Y Train']

        symbols = ['IBM']
        unalteredPrices = get_data(symbols,dates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        unalteredPrices= unalteredPrices/unalteredPrices.ix[0,:]

        ax = unalteredPrices.plot(title="Y Train/Price/Pred Y", label = "Price",color ='b')
        fiveDayPrices.plot(label = "Y Train",ax=ax,color = 'r')
        yPredXPriceDF.plot(label = "Predicted Y", ax=ax, color = "g")

        rmse = math.sqrt(((trainY - predYTrain) ** 2).sum()/trainY.shape[0])
        # print "in sample rmse"
        # print rmse
        c = np.corrcoef(predYTrain, y=trainY)
        # print "in sample corr: ", c[0,1]
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        plt.show()

        return learner,yPredXPriceDF,stats,unalteredPrices

    def setUpTestData(self,dates,learner, stats):
        momentumDF, actual5DayChange, unalteredPrices = self.getMomentum(dates)
        spyMomentum = self.getSPYMomentum(dates)
        stdDF=self.getVolatility(dates)

        SMA= pd.rolling_mean(unalteredPrices, window = 3)
        SMA= SMA.dropna()
        bb_value = (unalteredPrices-SMA)/(2*stdDF)

        bbMean = stats[4]
        bbStd = stats[5]
        bb_value = (bb_value-bbMean) / bbStd
        momentumMean = stats[0]
        momentumStd = stats[1]
        momentumDF = (momentumDF-momentumMean) / momentumStd
        spyMean = spyMomentum.mean()
        spyStd = spyMomentum.std()
        spyDF = (spyMomentum-spyMean)/ spyStd
        volatilityMean = stats[2]
        volatilityStd = stats[3]
        stdDF = (stdDF-volatilityMean)/ volatilityStd


        spyDF = spyDF[3:-3]
        momentumDF = momentumDF[3:-3]
        stdDF = stdDF[3:-3]
        actual5DayChange = actual5DayChange[3:-3]
        bb_value = bb_value[3:-3]
        unalteredPrices= unalteredPrices/unalteredPrices.ix[0,:]
        unalteredPrices = unalteredPrices[3:-3]

        actual5DayChange = actual5DayChange+1

        allDF = np.ones((momentumDF.shape[0],4))
        allDF[:,0]= momentumDF['IBM']
        allDF[:,1]= stdDF['IBM']
        allDF[:,2]= bb_value['IBM']
        allDF[:,3]= actual5DayChange['IBM']

        trainX = allDF[:,0:-1]
        trainY = allDF[:,-1]

        # print trainX
        predYTrain = learner.query(trainX) # get the predictions        sy = sknn.fit(trainX, trainY).predict(testX)
        # predYTest = learner.query(testX)


        #b3 predytrain is 972, future price df is 471
        yPredDF = pd.DataFrame(predYTrain, index = actual5DayChange.index)

        yPredXPriceDF= unalteredPrices*yPredDF.values
        fiveDayPrices = unalteredPrices * actual5DayChange.values

        yPredXPriceDF.columns = ['Predicted Y']
        fiveDayPrices.columns = ['Y Train']

        symbols = ['IBM']
        unalteredPrices = get_data(symbols,dates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        unalteredPrices= unalteredPrices/unalteredPrices.ix[0,:]

        ax = unalteredPrices.plot(title="Y Train/Price/Pred Y", label = "Price",color ='b')
        fiveDayPrices.plot(label = "Y Train",ax=ax,color = 'r')
        yPredXPriceDF.plot(label = "Pred Y", ax=ax, color = "g")

        rmse = math.sqrt(((trainY - predYTrain) ** 2).sum()/trainY.shape[0])
        # print "rmse"
        # print rmse
        c = np.corrcoef(predYTrain, y=trainY)
        # print "corr: ", c[0,1]
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        plt.show()

        return yPredXPriceDF
    def getVolatility(self,dates):
        # dates = pd.date_range('2007-12-31', '2009-12-31')
        symbols = ['IBM']
        unalteredPrices = get_data(symbols,dates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()

        daily_returns = unalteredPrices.copy()
        #print daily_returns
        daily_returns[1:] = (unalteredPrices[1:]/ unalteredPrices[:-1].values)-1
        daily_returns.ix[0,:]=0
        # print "printing daily returns"
        # print daily_returns
        # std = daily_returns.rolling_std()
        std = pd.rolling_std(daily_returns,5)
        # print "stndard dev of daily retunrs"
        # print std

        # print std
        return std

    def getMomentum(self,dates):
        symbols = ['IBM']
        forwardShiftedPrices = get_data(symbols,dates,addSPY=False)
        forwardShiftedPrices = forwardShiftedPrices.dropna()
        forwardShiftedPrices = forwardShiftedPrices.shift(-3)

        backwardShiftedPrices = get_data(symbols,dates,addSPY=False)
        backwardShiftedPrices = backwardShiftedPrices.dropna()
        backwardShiftedPrices = backwardShiftedPrices.shift(3)

        unalteredPrices = get_data(symbols,dates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()

        weekPercentChange = (forwardShiftedPrices/unalteredPrices)-1

        momentumDF = (unalteredPrices/backwardShiftedPrices)-1

        # print "whats the week"
        # print weekPercentChange

        return momentumDF,weekPercentChange,unalteredPrices

    def getSPYMomentum(self,dates):


        backwardShiftedPrices = get_data([],dates,addSPY=True)
        backwardShiftedPrices = backwardShiftedPrices.dropna()
        backwardShiftedPrices = backwardShiftedPrices.shift(3)

        unalteredPrices = get_data([],dates,addSPY=True)
        unalteredPrices = unalteredPrices.dropna()

        momentumDF = (unalteredPrices/backwardShiftedPrices)-1

        return momentumDF



    def assess_portfolio(self,prices, \
        allocs=[1], \
         rfr=0.0, sf=252.0, \
        sv = 10000):

        # Get daily portfolio value
        #print normed
        # normed = prices/prices.ix[0,:]
        # #print normed
        #
        # alloced = normed*allocs
        #
        # pos_vals = alloced * sv
        #
        # portfolio_val = pos_vals.sum(axis=1)
        portfolio_val = prices
        portfolio_val_returns = (prices/prices.shift(1))-1


        if portfolio_val.shape[0]>=1:
            cr = (portfolio_val.tail(1)/portfolio_val[0])-1
        adr = portfolio_val_returns.mean()
        sddr = portfolio_val_returns.std()
        #sr =sqrt(freq) * ( mean(daily_rets-daily_rf)/std(daily_rets))
        sr = (np.sqrt(sf)) * ((adr-rfr)/sddr)
        # Compare daily portfolio value with SPY using a normalized plot


        # Add code here to properly compute end value
        ev = sv+ (sv*cr)

        return cr, adr, sddr, sr
def compute_portvals(orders_file = "./Orders/orders.csv", start_val = 10000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    # orders_all = get_data(syms, dates)  # automatically adds SPY
    exceedsLeverage = True
    exceededDate = None
    ordersDF= pd.read_csv(orders_file, index_col = "Date", parse_dates = True, usecols = ['Date', 'Symbol','Order','Shares'])

    while exceedsLeverage==True:
        if exceededDate != None:
            # print "here is exceeded date"
            # print exceededDate
            if exceededDate in ordersDF:
                # print "exceeeded date is in there"

                if ordersDF.ix[exceededDate] is not None:
                    # ordersDF=ordersDF.drop([exceededDate])
                    # print "setting orders df to 0"
                    # print "made it past error"
                    ordersDF.ix[exceededDate, 'Shares']=0
            else:
                exceedsLeverage = False
        syms = pd.unique(ordersDF.Symbol.ravel())
        # print "type of syms"
        # print type(syms)
        syms = syms.tolist()
        # print syms
        #
        # print "dis what ordersDf looks like"
        # print ordersDF

        sd = ordersDF.index.min()
        ed = dt.date(2009, 12, 31)
        # Read in adjusted closing prices for given symbols, date range
        dates = pd.date_range(sd, ed)
        df1 = pd.DataFrame(index=dates)
        dfPrices = get_data(syms,dates,True)
        # print "df prices is the best in the building"
        dfPrices.loc[:,'Cash']=pd.Series(1,index=dfPrices.index)
        # print dfPrices
        # dfSPY = pd.read_csv("../data/SPY.csv", index_col = "Date", parse_dates = True, usecols = ['Date', 'Adj Close'])

        dfTrades = dfPrices.copy()
        dfTrades.ix[:] = 0
        # print "dftrades below"
        # print dfTrades

    #leverage = (sum(longs)+sum(abs(shorts))) / (sum(longs)- sum(abs(shorts)) + cash)
    #keep a running sum of longs and shorts
        #if leverage will go to >2.0, don't let trade happen.
        for index, row in ordersDF.iterrows():
            if row['Order'] == "BUY":
                dfTrades.loc[index, row['Symbol']] = float(dfTrades.loc[index, row['Symbol']])+ float(row['Shares'])
                dfTrades.loc[index, 'Cash'] = float(dfTrades.loc[index, 'Cash']) + float(row['Shares'] *-1 * dfPrices.loc[index,row['Symbol']])

            elif row['Order']== "SELL":
                dfTrades.loc[index, row['Symbol']] =float(dfTrades.loc[index, row['Symbol']])+ (-1 * float(row['Shares']))
                dfTrades.loc[index, 'Cash'] = float(dfTrades.loc[index, 'Cash']) + float(row['Shares'])  * float(dfPrices.loc[index,row['Symbol']])

            # if ordersDF.loc[index,row]
        # print dfTrades

        dfHoldings = dfTrades.copy()

        dfHoldings.ix[:] = 0

        #first row of dfHOldings = any shares bought on day 1. cash = start value - change in cash on day 1
        #all other rows of dfHoldings = shares from current day-1 + any change in current day
        for i in range (dfHoldings.shape[1]):
            dfHoldings.ix[0,i] = dfTrades.ix[0,i]
        dfHoldings.ix[0, dfHoldings.shape[1]-1] = start_val+float(dfTrades.ix[0,dfTrades.shape[1]-1])
        # print "dfHoldings below"
        # print dfHoldings


        dfHoldings[:] = dfTrades.cumsum()
        dfHoldings.ix[:,-1]= dfHoldings.ix[:,-1] + 10000

        dfValues = dfHoldings.copy()
        dfValues = dfHoldings* dfPrices

        # print "df values below"
        # print dfValues

    #leverage = (sum(longs)+sum(abs(shorts))) / (sum(longs)- sum(abs(shorts)) + cash)

        leverage = dfValues.copy()
        # leverage['leverage'] =  leverage[:,:-1].sum(axis=1)  /leverage.sum(axis=1)
        absoluteLeverage = dfValues.copy()
        allColumnsExceptCash = list(dfValues)
        allColumnsExceptCash.remove('Cash')

        absoluteLeverage.ix[:] = np.abs(absoluteLeverage.ix[:])
        # old ways leverage['leverage'] =  leverage[allColumns].sum(axis=1)/leverage.sum(axis=1)
        leverage['leverage'] =  absoluteLeverage[allColumnsExceptCash].sum(axis=1)/leverage.sum(axis=1)

        # print "leverage below post claculated"
        # print leverage
        exceededLeverage= leverage[np.abs(leverage.leverage)>2.0]
        # print "exceeeded leverage below"
        # print exceededLeverage

        #what if leverage is exceeded not by a trade, and then after that, while leverage is still exceeded,
        #another trade comes in.  should that trade be blocked? even if it's a sell?

        if exceededLeverage.shape[0]>0:
            # print "the first element of exceededlev"
            # print exceededLeverage.index[0]
            exceededDate = exceededLeverage.index[0]

        else:
            exceedsLeverage = False




    portfolio_val = dfValues.sum(axis=1)

    # print portfolio_val
    return portfolio_val

def compute_portvals_test(orders_file = "./Orders/orderstest.csv", start_val = 10000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    # orders_all = get_data(syms, dates)  # automatically adds SPY
    exceedsLeverage = True
    exceededDate = None
    ordersDF= pd.read_csv(orders_file, index_col = "Date", parse_dates = True, usecols = ['Date', 'Symbol','Order','Shares'])

    while exceedsLeverage==True:
        if exceededDate != None:
            # print "here is exceeded date"
            # print exceededDate
            if exceededDate in ordersDF:
                # print "exceeeded date is in there"

                if ordersDF.ix[exceededDate] is not None:
                    # ordersDF=ordersDF.drop([exceededDate])
                    # print "setting orders df to 0"
                    # print "made it past error"
                    ordersDF.ix[exceededDate, 'Shares']=0
            else:
                exceedsLeverage = False
        syms = pd.unique(ordersDF.Symbol.ravel())
        # print "type of syms"
        # print type(syms)
        syms = syms.tolist()
        # print syms
        #
        # print "dis what ordersDf looks like"
        # print ordersDF

        sd = ordersDF.index.min()
        ed = "2011-12-31"
        # Read in adjusted closing prices for given symbols, date range
        dates = pd.date_range(sd, ed)
        df1 = pd.DataFrame(index=dates)
        dfPrices = get_data(syms,dates,True)
        # print "df prices is the best in the building"
        dfPrices.loc[:,'Cash']=pd.Series(1,index=dfPrices.index)
        # print dfPrices
        # dfSPY = pd.read_csv("../data/SPY.csv", index_col = "Date", parse_dates = True, usecols = ['Date', 'Adj Close'])

        dfTrades = dfPrices.copy()
        dfTrades.ix[:] = 0
        # print "dftrades below"
        # print dfTrades

    #leverage = (sum(longs)+sum(abs(shorts))) / (sum(longs)- sum(abs(shorts)) + cash)
    #keep a running sum of longs and shorts
        #if leverage will go to >2.0, don't let trade happen.
        for index, row in ordersDF.iterrows():
            if row['Order'] == "BUY":
                dfTrades.loc[index, row['Symbol']] = float(dfTrades.loc[index, row['Symbol']])+ float(row['Shares'])
                dfTrades.loc[index, 'Cash'] = float(dfTrades.loc[index, 'Cash']) + float(row['Shares'] *-1 * dfPrices.loc[index,row['Symbol']])

            elif row['Order']== "SELL":
                dfTrades.loc[index, row['Symbol']] =float(dfTrades.loc[index, row['Symbol']])+ (-1 * float(row['Shares']))
                dfTrades.loc[index, 'Cash'] = float(dfTrades.loc[index, 'Cash']) + float(row['Shares'])  * float(dfPrices.loc[index,row['Symbol']])

            # if ordersDF.loc[index,row]
        # print dfTrades

        dfHoldings = dfTrades.copy()

        dfHoldings.ix[:] = 0

        #first row of dfHOldings = any shares bought on day 1. cash = start value - change in cash on day 1
        #all other rows of dfHoldings = shares from current day-1 + any change in current day
        for i in range (dfHoldings.shape[1]):
            dfHoldings.ix[0,i] = dfTrades.ix[0,i]
        dfHoldings.ix[0, dfHoldings.shape[1]-1] = start_val+float(dfTrades.ix[0,dfTrades.shape[1]-1])
        # print "dfHoldings below"
        # print dfHoldings


        dfHoldings[:] = dfTrades.cumsum()
        dfHoldings.ix[:,-1]= dfHoldings.ix[:,-1] + 10000

        dfValues = dfHoldings.copy()
        dfValues = dfHoldings* dfPrices

        # print "df values below"
        # print dfValues

    #leverage = (sum(longs)+sum(abs(shorts))) / (sum(longs)- sum(abs(shorts)) + cash)

        leverage = dfValues.copy()
        # leverage['leverage'] =  leverage[:,:-1].sum(axis=1)  /leverage.sum(axis=1)
        absoluteLeverage = dfValues.copy()
        allColumnsExceptCash = list(dfValues)
        allColumnsExceptCash.remove('Cash')

        absoluteLeverage.ix[:] = np.abs(absoluteLeverage.ix[:])
        # old ways leverage['leverage'] =  leverage[allColumns].sum(axis=1)/leverage.sum(axis=1)
        leverage['leverage'] =  absoluteLeverage[allColumnsExceptCash].sum(axis=1)/leverage.sum(axis=1)

        # print "leverage below post claculated"
        # print leverage
        exceededLeverage= leverage[np.abs(leverage.leverage)>2.0]
        # print "exceeeded leverage below"
        # print exceededLeverage

        #what if leverage is exceeded not by a trade, and then after that, while leverage is still exceeded,
        #another trade comes in.  should that trade be blocked? even if it's a sell?

        if exceededLeverage.shape[0]>0:
            # print "the first element of exceededlev"
            # print exceededLeverage.index[0]
            exceededDate = exceededLeverage.index[0]

        else:
            exceedsLeverage = False




    portfolio_val = dfValues.sum(axis=1)

    # print portfolio_val
    return portfolio_val

l = learner()
dates = pd.date_range('2007-12-31', '2009-12-31')

learner,data, stats,unalteredPrices = l.setUp(dates)
l.trade(data,unalteredPrices)
portvals = compute_portvals()
portvals.columns = ["Portfolio"]

# portvals.title = "Portfolio"
stock = get_data(["IBM"],dates,addSPY=False)
stock = stock.dropna()
stock = (stock / stock.ix[0,:])*10000

ax = stock.plot(title = "Daily Portfolio Value", mark_right = False)
ax.set_xlabel("Date")
ax.set_ylabel("Normalized price")

portvals.plot(label = "Portfolio", ax=ax,color = 'r')
plt.show()
cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = l.assess_portfolio(portvals)
# print "Date Range: {} to {}".format(start_date, end_date)
# print
# print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
# print
# print "Cumulative Return of Fund: {}".format(cum_ret)
# print
# print "Standard Deviation of Fund: {}".format(std_daily_ret)
# print
# print "Average Daily Return of Fund: {}".format(avg_daily_ret)
# print
# print "Final Portfolio Value: {}".format(portvals[-1])
#
# print "Length of dataframe is : {}".format(len(portvals))

testDates=  pd.date_range('2009-12-31', '2011-12-31')

testData = l.setUpTestData(testDates,learner,stats)
l.tradeTest(testData)

portvalsTest = compute_portvals_test()

portvalsTest.columns = ['Out Sample Portfolio']
# print "columns of portval"
# print portvalsTest.columns
stock = get_data(["IBM"],testDates,addSPY=False)
stock = stock.dropna()

stock = (stock / stock.ix[0,:])*10000
axTest = stock.plot(title = "Daily Portfolio Value-Out Sample", mark_right = False)

portvalsTest.plot(label = 'Out Sample Portfolio',ax=axTest,color = 'r')

axTest.set_xlabel("Date")
axTest.set_ylabel("Normalized price")
plt.show()
cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = l.assess_portfolio(portvalsTest)

print "-----***** begin out of sample ****-----"
print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
print
print "Cumulative Return of Fund: {}".format(cum_ret)
print
print "Standard Deviation of Fund: {}".format(std_daily_ret)
print
print "Average Daily Return of Fund: {}".format(avg_daily_ret)
print
print "Final Portfolio Value: {}".format(portvals[-1])

print "Length of dataframe is : {}".format(len(portvalsTest))