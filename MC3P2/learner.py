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
        headerColumns = ["Date","Symbol","Order","Shares"]
        writer.writerow(headerColumns)

        daysHeldFor = -1

        unalteredPrices = unalteredPrices[3:-3]

        for i in data.index.date:
            #Always hold position for at least 5 days
            if daysHeldFor>=0 and daysHeldFor < 5:
                daysHeldFor +=1
                continue
            else:

                if (data["Predicted Y"].ix[i]> unalteredPrices["IBM"].ix[i] and (longIsOpen==False or shortIsOpen==True)):
                    #Close the short by buying out the position.
                    orderType, shares, symbol, orderDate, daysHeldFor = "BUY","100","IBM", str(i),0
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

                elif data["Predicted Y"].ix[i]<unalteredPrices["IBM"].ix[i] and (shortIsOpen==False or longIsOpen==True):
                    #Close the long position by selling.
                    orderType, shares, symbol, orderDate, daysHeldFor = "SELL","100","IBM", str(i),0
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
        self.displayEntryExitChart(buys, close, shorts)

    def displayEntryExitChart(self, buys, close, shorts):
        symbols = ['IBM']
        unalteredPrices = get_data(symbols, dates, addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        unalteredPrices = unalteredPrices / unalteredPrices.ix[0, :]

        ax = unalteredPrices["IBM"].plot(title="Entry/Exit Graph", label="IBM", color='b')

        ymin, ymax = ax.get_ylim()

        plt.vlines(buys, ymin=ymin, ymax=ymax, color='g', label='Buys')
        plt.vlines(close, ymin=ymin, ymax=ymax, color='k', label='Exits')
        plt.vlines(shorts, ymin=ymin, ymax=ymax, color='r', label='Shorts')

        plt.show()

    def setUp(self,dates):
        fiveDayPriceChange, bollingerBandValues, momentumDF, stats, unalteredPrices, volatilityDF = self.calculateTrainingStats(
            dates)

        #shave off the NA's in the first three and last 3 rows.  Necessary because the stats
        #are calculated on a rolling basis.
        fiveDayPriceChange, trainX, trainY, unalteredPrices = self.prepareTrainXandY(bollingerBandValues,
                                                                                     fiveDayPriceChange, momentumDF,
                                                                                     unalteredPrices, volatilityDF)

        # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
        learner = knn.KNNLearner(2,verbose = True) # create a knn learner
        learner.addEvidence(trainX, trainY) # train it

        fiveDayPrices, unalteredPrices, yPredTimesPriceDF = self.setYFromTrainingAndGetActualY(dates,
                                                                                               fiveDayPriceChange,
                                                                                               learner, trainX,
                                                                                               unalteredPrices)
        self.showChartYTrainYPred(fiveDayPrices, unalteredPrices, yPredTimesPriceDF)

        return learner,yPredTimesPriceDF,stats,unalteredPrices

    def showChartYTrainYPred(self, fiveDayPrices, unalteredPrices, yPredTimesPriceDF):
        ax = unalteredPrices.plot(title="Y Train/Price/Pred Y", label="Price", color='b')
        fiveDayPrices.plot(label="Y Train", ax=ax, color='r')
        yPredTimesPriceDF.plot(label="Predicted Y", ax=ax, color="g")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        plt.show()

    def prepareTrainXandY(self, bollingerBandValues, fiveDayPriceChange, momentumDF, unalteredPrices, volatilityDF):
        momentumDF = momentumDF[3:-3]
        volatilityDF = volatilityDF[3:-3]
        fiveDayPriceChange = fiveDayPriceChange[3:-3]
        bollingerBandValues = bollingerBandValues[3:-3]
        unalteredPrices = unalteredPrices / unalteredPrices.ix[0, :]
        unalteredPrices = unalteredPrices[3:-3]
        fiveDayPriceChange = fiveDayPriceChange + 1
        allDF = np.ones((momentumDF.shape[0], 4))
        allDF[:, 0] = momentumDF['IBM']
        allDF[:, 1] = volatilityDF['IBM']
        allDF[:, 2] = bollingerBandValues['IBM']
        allDF[:, 3] = fiveDayPriceChange['IBM']
        # The inputs
        trainX = allDF[:, 0:-1]
        # The outcome
        trainY = allDF[:, -1]
        return fiveDayPriceChange, trainX, trainY, unalteredPrices

    def calculateTrainingStats(self, dates):
        bollingerBandValue, fiveDayPriceChange, momentumDF, unalteredPrices, volatilityDF = self.beginParameterCalculations(
            dates)

        bollingerBandMean = bollingerBandValue.mean()
        bollingerBandStandardDeviation = bollingerBandValue.std()
        bollingerBandValue = (bollingerBandValue - bollingerBandMean) / bollingerBandStandardDeviation

        momentumMean = momentumDF.mean()
        momentumStd = momentumDF.std()
        momentumDF = (momentumDF - momentumMean) / momentumStd

        volatilityMean = volatilityDF.mean()
        volatilityStd = volatilityDF.std()
        volatilityDF = (volatilityDF - volatilityMean) / volatilityStd

        stats = []
        stats.append(momentumMean)
        stats.append(momentumStd)
        stats.append(volatilityMean)
        stats.append(volatilityStd)
        stats.append(bollingerBandMean)
        stats.append(bollingerBandStandardDeviation)

        return fiveDayPriceChange, bollingerBandValue, momentumDF, stats, unalteredPrices, volatilityDF

    def beginParameterCalculations(self, dates):
        momentumDF, fiveDayPriceChange, unalteredPrices = self.getMomentum(dates)
        volatilityDF = self.getVolatility(dates)
        movingAverage = pd.rolling_mean(unalteredPrices, window=3)
        movingAverage = movingAverage.dropna()
        bollingerBandValue = (unalteredPrices - movingAverage) / (2 * volatilityDF)
        return bollingerBandValue, fiveDayPriceChange, momentumDF, unalteredPrices, volatilityDF

    def setUpTestData(self,dates,learner, stats):
        bollingerBandValue, fiveDayPriceChange, momentumDF, unalteredPrices, volatilityDF = self.beginParameterCalculations(
            dates)

        momentumMean,momentumStd,volatilityMean,volatilityStd,bbMean,bbStd = stats[0],stats[1],stats[2],stats[3],stats[4],stats[5]

        bollingerBandValues = (bollingerBandValue-bbMean) / bbStd
        momentumDF = (momentumDF-momentumMean) / momentumStd
        volatilityDF = (volatilityDF-volatilityMean)/ volatilityStd

        fiveDayPriceChange, trainX, trainY, unalteredPrices = self.prepareTrainXandY(bollingerBandValues,
                                                                                     fiveDayPriceChange, momentumDF,
                                                                                     unalteredPrices, volatilityDF)

        fiveDayPrices, unalteredPrices, yPredTimesPriceDF = self.setYFromTrainingAndGetActualY(dates,
                                                                                               fiveDayPriceChange,
                                                                                               learner, trainX,
                                                                                               unalteredPrices)

        self.showChartYTrainYPred(fiveDayPrices, unalteredPrices, yPredTimesPriceDF)

        return yPredTimesPriceDF

    def setYFromTrainingAndGetActualY(self, dates, fiveDayPriceChange, learner, trainX, unalteredPrices):
        predictedYFromTraining = learner.query(
            trainX)  # get the predictions        sy = sknn.fit(trainX, trainY).predict(testX)
        yPredictedDF = pd.DataFrame(predictedYFromTraining, index=fiveDayPriceChange.index)
        yPredTimesPriceDF = yPredictedDF.values * unalteredPrices
        fiveDayPrices = fiveDayPriceChange.values * unalteredPrices
        yPredTimesPriceDF.columns = ['Predicted Y']
        fiveDayPrices.columns = ['Y Train']
        symbols = ['IBM']
        unalteredPrices = get_data(symbols, dates, addSPY=False)
        unalteredPrices = unalteredPrices.dropna()
        unalteredPrices = unalteredPrices / unalteredPrices.ix[0, :]
        return fiveDayPrices, unalteredPrices, yPredTimesPriceDF

    def getVolatility(self,dates):
        # dates = pd.date_range('2007-12-31', '2009-12-31')
        symbols = ['IBM']
        unalteredPrices = get_data(symbols,dates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()

        daily_returns = unalteredPrices.copy()

        daily_returns[1:] = (unalteredPrices[1:]/ unalteredPrices[:-1].values)-1
        daily_returns.ix[0,:]=0

        std = pd.rolling_std(daily_returns,5)

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
        startValue = 10000):

        portfolio_val = prices
        portfolio_val_returns = (prices/prices.shift(1))-1

        if portfolio_val.shape[0]>=1:
            cumulativeReturn = (portfolio_val.tail(1)/portfolio_val[0])-1
        meanOfDailyReturns = portfolio_val_returns.mean()
        standardDeviationOfDailyReturns = portfolio_val_returns.std()
        sharpeRatio = (np.sqrt(sf)) * ((meanOfDailyReturns-rfr)/standardDeviationOfDailyReturns)
        # Compare daily portfolio value with SPY using a normalized plot

        return cumulativeReturn, meanOfDailyReturns, standardDeviationOfDailyReturns, sharpeRatio
def compute_portvals(orders_file = "./Orders/orders.csv", start_val = 10000, endDate = dt.date(2009, 12, 31)):
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

        startDate = ordersDF.index.min()
        # Read in adjusted closing prices for given symbols, date range
        dates = pd.date_range(startDate, endDate)
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

unalteredTestPrices = get_data(['IBM'], testDates,addSPY=False).dropna()
unalteredTestPrices = unalteredTestPrices / unalteredTestPrices.ix[0,:]
l.trade(testData, unalteredTestPrices)

portvalsTest = compute_portvals(endDate=dt.date(2011,12,31))

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