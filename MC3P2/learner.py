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

#improvements
#1) unaltered prices is created and used and passed in many different places.  probably easiest to just calculate it once and pass it in when needed.
#2) Get the legends correct.  Every line in legend and every line labeled
#3) A lot of methods do multiple things.  limit them to one thing per method

class learner(object):

    #Creates a file of trades based on daily prices and what the strategy says to do.
    def trade(self,strategyData,dailyPrices,symbol):
        longIsOpen = False
        shortIsOpen = False
        buys = []
        shorts = []
        close = []

        fileName = "./Orders/orders.csv"
        f=open(fileName,"w+")
        writer = csv.writer(f)
        headerColumns = ["Date","Symbol","Order","Shares"]
        writer.writerow(headerColumns)

        daysHeldFor = -1

        dailyPrices = dailyPrices[3:-3]

        for i in strategyData.index.date:
            #Always hold position for at least 5 days
            if daysHeldFor>=0 and daysHeldFor < 5:
                daysHeldFor +=1
                continue
            else:

                if (strategyData["Predicted Y"].ix[i]> dailyPrices[symbol].ix[i] and (longIsOpen==False or shortIsOpen==True)):
                    #Close the short by buying out the position.
                    orderType, shares, symbol, orderDate, daysHeldFor = "BUY","100",symbol, str(i),0
                    rowValues = [orderDate,symbol,orderType,shares]
                    writer.writerow(rowValues)
                    if shortIsOpen == True:
                        shortIsOpen =False
                        close.append(i)
                    #Open a long position
                    else:
                        # print "tim eto buy"
                        buys.append(i)
                        longIsOpen = True

                elif strategyData["Predicted Y"].ix[i]<dailyPrices[symbol].ix[i] and (shortIsOpen==False or longIsOpen==True):
                    #Close the long position by selling.
                    orderType, shares, symbol, orderDate, daysHeldFor = "SELL","100",symbol, str(i),0
                    rowValues = [orderDate,symbol,orderType,shares]
                    writer.writerow(rowValues)
                    if longIsOpen == True:
                        close.append(i)
                        longIsOpen=False
                    #Open a short position.
                    else:
                        shorts.append(i)
                        shortIsOpen = True
        f.close()

        #The below code will display the entry/exit graph
        self.displayEntryExitChart(buys, close, shorts,symbol)

    #Create and display a chart depicting when a position begins and ends.
    def displayEntryExitChart(self, buys, close, shorts,symbol):
        symbols = [symbol]
        unalteredPrices = get_data(symbols, dates, addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        unalteredPrices = unalteredPrices / unalteredPrices.ix[0, :]

        ax = unalteredPrices[symbol].plot(title="Entry/Exit Graph", label=symbol, color='b')

        ymin, ymax = ax.get_ylim()

        plt.vlines(buys, ymin=ymin, ymax=ymax, color='g', label='Buys')
        plt.vlines(close, ymin=ymin, ymax=ymax, color='k', label='Exits')
        plt.vlines(shorts, ymin=ymin, ymax=ymax, color='r', label='Shorts')

        plt.show()

    def normalizeDataFrame(self,valuesDF):
        return ((valuesDF - valuesDF.mean()) / valuesDF.std())

    def showChartYTrainYPred(self, fiveDayPrices, dailyPrices, yPredTimesPriceDF):
        ax = dailyPrices.plot(title="Y Train/Price/Pred Y", label="Price", color='b')
        fiveDayPrices.plot(label="Y Train", ax=ax, color='r')
        yPredTimesPriceDF.plot(label="Predicted Y", ax=ax, color="g")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        plt.show()

    #Cleans up the inputs and makes them a consistent size that each of them have values for and puts everything into a single Training X dataframe and Training Y dataframe.
    def prepareTrainXandY(self, bollingerBandValues, fiveDayPriceChange, momentumDF, unalteredPrices, volatilityDF,symbol):
        momentumDF = momentumDF[3:-3]
        volatilityDF = volatilityDF[3:-3]
        fiveDayPriceChange = fiveDayPriceChange[3:-3]
        bollingerBandValues = bollingerBandValues[3:-3]
        unalteredPrices = unalteredPrices / unalteredPrices.ix[0, :]
        unalteredPrices = unalteredPrices[3:-3]
        fiveDayPriceChange = fiveDayPriceChange + 1
        allDF = np.ones((momentumDF.shape[0], 4))
        allDF[:, 0] = momentumDF[symbol]
        allDF[:, 1] = volatilityDF[symbol]
        allDF[:, 2] = bollingerBandValues[symbol]
        allDF[:, 3] = fiveDayPriceChange[symbol]
        # The inputs
        trainX = allDF[:, 0:-1]
        # The outcome
        trainY = allDF[:, -1]
        return fiveDayPriceChange, trainX, trainY, unalteredPrices

    def getStats(self,momentumDf, volatilityDf, bollingerDf):
        stats = []
        stats.append(momentumDf.mean())
        stats.append(momentumDf.std())
        stats.append(volatilityDf.mean())
        stats.append(volatilityDf.std())
        stats.append(bollingerDf.mean())
        stats.append(bollingerDf.std())
        return stats

    def getBollingerBandVAlue(self,symbol, dates, volatilityDF):
        unalteredPrices = get_data([symbol],dates,addSPY=False).dropna()
        movingAverage = pd.rolling_mean(unalteredPrices, window=3)
        movingAverage = movingAverage.dropna()
        bollingerBandValue = (unalteredPrices - movingAverage) / (2 * volatilityDF)
        return bollingerBandValue

    #Calculate testing data, normalize it based on mean and standard deviation, use training data to calculate predictions.
    def setUpTestData(self,dates,learner, stats,symbol):
        unalteredPrices = get_data([symbol], dates, addSPY=False).dropna()
        momentumDF = self.getMomentum(dates,symbol)
        fiveDayPriceChange = self.getWeekPercentPriceChange(dates,symbol)
        volatilityDF = self.getVolatility(dates,symbol)
        bollingerBandValue = self.getBollingerBandVAlue(symbol,dates,volatilityDF)


        momentumMean,momentumStd,volatilityMean,volatilityStd,bbMean,bbStd = stats[0],stats[1],stats[2],stats[3],stats[4],stats[5]

        bollingerBandValues = (bollingerBandValue-bbMean) / bbStd
        momentumDF = (momentumDF-momentumMean) / momentumStd
        volatilityDF = (volatilityDF-volatilityMean)/ volatilityStd

        fiveDayPriceChange, trainX, trainY, cleanedDailyPrices = self.prepareTrainXandY(bollingerBandValues,
                                                                                     fiveDayPriceChange, momentumDF,
                                                                                     unalteredPrices, volatilityDF,symbol)

        fiveDayPrices, normalizedDailyPrices, yPredTimesPriceDF = self.setYFromTrainingAndGetActualY(dates,
                                                                                               fiveDayPriceChange,
                                                                                               learner, trainX,
                                                                                               cleanedDailyPrices,symbol)

        self.showChartYTrainYPred(fiveDayPrices, normalizedDailyPrices, yPredTimesPriceDF)

        return yPredTimesPriceDF



    #Use learner to get what the learner thinks price will be
    def setYFromTrainingAndGetActualY(self, dates, fiveDayPriceChange, learner, trainX, unalteredPrices,symbol):
        predictedYFromTraining = learner.query(
            trainX)  # get the predictions        sy = sknn.fit(trainX, trainY).predict(testX)
        yPredictedDF = pd.DataFrame(predictedYFromTraining, index=fiveDayPriceChange.index)
        yPredTimesPriceDF = yPredictedDF.values * unalteredPrices
        fiveDayPrices = fiveDayPriceChange.values * unalteredPrices
        yPredTimesPriceDF.columns = ['Predicted Y']
        fiveDayPrices.columns = ['Y Train']
        symbols = [symbol]
        unalteredPrices = get_data(symbols, dates, addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        normalizedDailyPrices = unalteredPrices / unalteredPrices.ix[0, :]
        return fiveDayPrices, normalizedDailyPrices, yPredTimesPriceDF

    def getVolatility(self,dates,symbol):
        daily_returns = get_data([symbol],dates,addSPY=False).dropna()

        daily_returns[1:] = (daily_returns[1:]/ daily_returns[:-1].values)-1
        daily_returns.ix[0,:]=0

        std = pd.rolling_std(daily_returns,5)

        return std

    def getMomentum(self,dates,symbol):
        symbols = [symbol]
        forwardShiftedPrices = get_data(symbols,dates,addSPY=False).dropna().shift(-3)
        backwardShiftedPrices = get_data(symbols,dates,addSPY=False).dropna().shift(3)
        unalteredPrices = get_data(symbols,dates,addSPY=False).dropna()

        momentumDF = (unalteredPrices/backwardShiftedPrices)-1

        return momentumDF

    def getWeekPercentPriceChange(self,dates,symbol):
        forwardShiftedPrices = get_data([symbol],dates,addSPY=False).dropna()
        forwardShiftedPrices = forwardShiftedPrices.shift(-3)
        unalteredPrices = get_data([symbol],dates,addSPY=False).dropna()
        return (forwardShiftedPrices/unalteredPrices)-1


    def getSPYMomentum(self,dates,unalteredPrices):
        backwardShiftedPrices = get_data([],dates,addSPY=True)
        backwardShiftedPrices = backwardShiftedPrices.dropna()
        backwardShiftedPrices = backwardShiftedPrices.shift(3)

        momentumDF = (unalteredPrices/backwardShiftedPrices)-1

        return momentumDF

    def assess_portfolio(self,prices, \
        allocs=[1], \
         rfr=0.0, sf=252.0, \
        startValue = 10000):

        portfolio_val = prices
        portfolio_val_returns = (prices/prices.shift(1))-1

        if portfolio_val.shape[0]>=1:
            cumulativeReturn = (portfolio_val.tail(1)/portfolio_val[0])-1
        meanOfDailyReturns = portfolio_val_returns.mean()
        standardDeviationOfDailyReturns = portfolio_val_returns.std()
        sharpeRatio = (np.sqrt(sf)) * ((meanOfDailyReturns-rfr)/standardDeviationOfDailyReturns)

        return cumulativeReturn, meanOfDailyReturns, standardDeviationOfDailyReturns, sharpeRatio
def compute_portvals(orders_file = "./Orders/orders.csv", start_val = 10000, endDate = dt.date(2009, 12, 31)):
    exceededDate = None
    ordersDF= pd.read_csv(orders_file, index_col = "Date", parse_dates = True, usecols = ['Date', 'Symbol','Order','Shares'])

    if exceededDate != None:
        if exceededDate in ordersDF:
            if ordersDF.ix[exceededDate] is not None:
                ordersDF.ix[exceededDate, 'Shares']=0
        else:
            exceedsLeverage = False
    syms = pd.unique(ordersDF.Symbol.ravel())

    syms = syms.tolist()

    startDate = ordersDF.index.min()
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(startDate, endDate)
    dfPrices = get_data(syms,dates,True)
    dfPrices.loc[:,'Cash']=pd.Series(1,index=dfPrices.index)

    dfTrades = dfPrices.copy()
    dfTrades.ix[:] = 0

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

    dfHoldings[:] = dfTrades.cumsum()
    dfHoldings.ix[:,-1]= dfHoldings.ix[:,-1] + 10000

    dfValues = dfHoldings* dfPrices

    portfolio_val = dfValues.sum(axis=1)

    return portfolio_val

