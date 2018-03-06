import StrategyLearner as sl
import numpy as np
import QLearner as ql
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time
import scipy.optimize as sco
from util import get_data, plot_data

class StrategyLearner(object):

    def __init__(self, \
        verbose = True):
        self.QLearner = ql.QLearner(num_states=(9992), \
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False)
        self.endtime = time.time() + 24
        self.position = 'NEUTRAL'


        pass


    def addEvidence(self,symbol = "IBM", startDate=dt.datetime(2007,12,31), endDate=dt.datetime(2009,12,31), sv = 10000):

        #Get a data frame of dates and closing prices from start date to end date
        dates = pd.date_range(startDate, endDate)

        #setUp will get relevant metrics such as momentum, volatility, etc into a single dataframe.
        allDataMetricsDF = self.setUp(dates,symbol)

        # print "initial size post set up. maybe set up mutates it"

        allDataMetricsDF = allDataMetricsDF[:,:]

        discretizedDataMetricsDF = self.discretize(allDataMetricsDF)

        shape = allDataMetricsDF.shape

        discretizedDataMetricsDF = discretizedDataMetricsDF.loc[1:,:]

        for iteration in range(0,200):
            if time.time() >=self.endtime:
                break

            steps=2
            robopos = 0

            datum = discretizedDataMetricsDF.iloc[4,:]
            #convert the very first location to a state.  start at 4th date so can use the preceeding 3 dates as well
            #to map out a state.
            state = self.makeIntoState(datum,robopos)
            action = self.QLearner.querysetstate(state) #set the state and get first action
            cumulativeRewards = 0
            r=0
            #iterate from the 3rd date until the last date
            for i in range (3,shape[0]-1):
                newpos = self.movebot(robopos,action)

                #move to new location according to action and then get a new action
                if newpos == 1:   #Buy
                    r = (allDataMetricsDF[i,-1]-1)
                    cumulativeRewards+=r
                elif newpos == 2: #Sell
                    r = (allDataMetricsDF[i,-1]-1)*-1
                    cumulativeRewards+=r
                else:
                    r = 0

                robopos = newpos
                steps+=1
                continue

                datum = discretizedDataMetricsDF.iloc[i,:]
                state = self.makeIntoState(datum,newpos)
                action = self.QLearner.query(state,r)

                robopos = action
                steps += 1

    def makeIntoState(self,data,currentPos):
        dataStr = str(int(currentPos))
        dataStr += str(int(data.iloc[0]))

        dataStr+=str(int(data.iloc[1]))
        dataStr+=str(int(data.iloc[2]))

        dataStr = int(dataStr)

        #Hash to the number of possible states.
        dataStr = dataStr% 250

        return dataStr

    def testPolicy(self,symbol = "IBM", startDate=dt.datetime(2007,12,31), endDate=dt.datetime(2009,12,31), sv = 10000, isPortfolio = True):

        self.QLearner.RandomActionRate = 0
        dates = pd.date_range(startDate, endDate)
        allDf = self.setUp(dates,symbol)

        allDf = allDf[:,:]

        data = self.discretizeTest(allDf)

        shape = allDf.shape

        zeros = np.zeros((shape[0],1))
        dfTrades = pd.DataFrame(zeros)
        portfolioND = np.zeros((shape[0],4))
        portfolio = pd.DataFrame(portfolioND)

        steps = 0
        robopos = 0

        datum = data.iloc[0,:] #convert the location to a state

        state = self.makeIntoState(datum,robopos)
        action = self.QLearner.querysetstate(state) #set the state and get first action
        cumulativeRewards = 0
        r=0

        holdings = 0
        portfolioCount=0
        for i in range (0,shape[0]-1):
            newpos = self.movebot(robopos,action)
            if i >0:# shape[0]-1:
                if newpos == 1:
                    r = allDf[i,-1]-1
                elif newpos == 2:
                    r = (allDf[i,-1]-1)*-1
                else:
                    r = 0
                cumulativeRewards+=r

            datum = data.iloc[i,:]
            state = self.makeIntoState(datum,newpos)
            action = self.QLearner.querysetstate(state)

            if holdings == 0 and newpos==1:
                holdings=100
            elif holdings ==0 and newpos ==2:
                holdings = -100
            elif holdings == 100 and newpos==1:
                holdings = 100
            elif holdings == 100 and newpos ==2:
                holdings = -100
            elif holdings == 100 and newpos ==0:
                holdings = 0
            elif holdings == -100 and newpos ==0:
                holdings= 0
            elif holdings == -100 and newpos == 1:
                holdings = 100
            elif holdings == -100 and newpos ==2:
                holdings = -100

            if robopos == action:
                dfTrades.iloc[i] = 0

            elif robopos == 1 and action == 2:
                dfTrades.iloc[i] = -200
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = symbol
                portfolio.loc[portfolioCount,2] = "SELL"
                portfolio.loc[portfolioCount,3] = -200
                portfolioCount+=1
            elif robopos == 1 and action == 0:
                dfTrades.iloc[i] = -100
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = symbol
                portfolio.loc[portfolioCount,2] = "SELL"
                portfolio.loc[portfolioCount,3] = -100
                portfolioCount+=1

            elif robopos == 2 and action == 1:
                dfTrades.iloc[i] = 200
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = symbol
                portfolio.loc[portfolioCount,2] = "BUY"
                portfolio.loc[portfolioCount,3] = 200
                portfolioCount+=1

            elif robopos == 2 and action == 0:
                dfTrades.iloc[i] = 100
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = symbol
                portfolio.loc[portfolioCount,2] = "BUY"
                portfolio.loc[portfolioCount,3] = 100
                portfolioCount+=1

            elif robopos == 0 and action == 1:
                dfTrades.iloc[i] = 100
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = symbol
                portfolio.loc[portfolioCount,2] = "BUY"
                portfolio.loc[portfolioCount,3] = 100
                portfolioCount+=1

            elif robopos == 0 and action == 2:
                dfTrades.iloc[i] = -100
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = symbol
                portfolio.loc[portfolioCount,2] = "SELL"
                portfolio.loc[portfolioCount,3] = -100
                portfolioCount+=1

            robopos = action
            steps += 1
        if (isPortfolio):
            return portfolio
        else:
            return dfTrades

    def discretizeTest(self,data):
        data = data[1:,:]
        data = pd.DataFrame(data)

        nd1 = data.ix[:,0]
        nd2 = data.ix[:,1]
        nd3= data.ix[:,2]

        bins1 = self.thresholds0[1]
        bins2= self.thresholds1[1]
        bins3=self.thresholds2[1]

        cut1=np.digitize(nd1,bins1) -1
        cut2= np.digitize(nd2,bins2) -1
        cut3= np.digitize(nd3,bins3) -1

        cut1[cut1>9] = 9
        cut1[cut1<0] = 0
        cut2[cut2>9] = 9
        cut2[cut2<0] = 0
        cut3[cut3>9] = 9
        cut3[cut3<0] = 0

        df1 = pd.DataFrame(cut1)
        df2 = pd.DataFrame(cut2)
        df3= pd.DataFrame(cut3)

        df = pd.concat([df1,df2],axis=1)
        df = pd.concat([df,df3],axis=1)

        return df


    def discretize(self,data):
        dataFrame= pd.DataFrame(data)
        dataFrame= dataFrame.ix[1:,:]

        #discretize the columns of the dataframe into 10 buckets.
        self.thresholds0 = pd.qcut(dataFrame.ix[:,0],10,labels=False,retbins=True)
        self.thresholds1 = pd.qcut(dataFrame.ix[:,1],10,labels=False,retbins=True)
        self.thresholds2 = pd.qcut(dataFrame.ix[:,2],10,labels=False,retbins=True)

        #convert the thresholds into their own dataframes
        thresholds0DF= pd.DataFrame(self.thresholds0[0])
        thresholds1DF= pd.DataFrame(self.thresholds1[0])
        thresholds2DF= pd.DataFrame(self.thresholds2[0])

        allDF = thresholds0DF.join(thresholds1DF, how ='inner')
        allDF = allDF.join(thresholds2DF, how = "inner")

        return allDF

    #Gets momentum, volatility, spy momentum, and bb values.  puts them into a single ndarray.  also puts averages and
    # standard deviations into a stats list variable for use later in normalization
    def setUp(self,dates,symbols):
        momentumDF, actual5DayChange, unalteredPrices = self.getMomentum(dates,symbols)
        volatilityDf=self.getVolatility(dates,symbols)

        SMA= pd.rolling_mean(unalteredPrices, window = 3)
        SMA= SMA.dropna()
        bollingerband_Value = (unalteredPrices-SMA)/(2*volatilityDf)

        actual5DayChange = actual5DayChange+1

        allDF = np.ones((momentumDF.shape[0],4))
        allDF[:,0]= momentumDF[symbols]
        allDF[:,1]= volatilityDf[symbols]
        allDF[:,2]= bollingerband_Value[symbols]
        allDF[:,3]= actual5DayChange[symbols]

        return allDF

    def getMomentum(self,dates,symbol):
        symbols = [symbol]
        forwardShiftedPrices = get_data(symbols,dates,addSPY=False)
        forwardShiftedPrices = forwardShiftedPrices.dropna()
        forwardShiftedPrices = forwardShiftedPrices.shift(-1)

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

    def getVolatility(self,dates,symbols):
        symbols = [symbols]
        unalteredPrices = get_data(symbols,dates,addSPY=False)
        unalteredPrices = unalteredPrices.dropna()

        pricesWithoutNa = unalteredPrices.copy()
        pricesWithoutNa[1:] = (unalteredPrices[1:]/ unalteredPrices[:-1].values)-1
        pricesWithoutNa.ix[0,:]=0

        volatility = pd.rolling_std(pricesWithoutNa,5)
        return volatility

    def movebot(self,oldpos,a):
        #a == 0 means hold, 1 means buy
        if a !=oldpos:
            oldpos = a

        newpos = oldpos
        return newpos

    def computeDailyPortfolioValues(self,ordersDataFrame,dateRange,symbol,start_val = 1):
        exceedsLeverage = False
        exceededDate = None
        prices_all = get_data([symbol], dateRange)
        ordersDataFrame = self.convertIndicesToDates(ordersDataFrame, prices_all)

        #If leverage is exceeded, then keep looping until it doesn't exceed leverage.
        while True:
            if exceededDate != None:
                if exceededDate in ordersDataFrame:
                    if ordersDataFrame.ix[exceededDate] is not None:
                        ordersDataFrame.ix[exceededDate, 'Shares']=0
                else:
                    exceedsLeverage = False

            # Read in adjusted closing prices for given symbols, date range
            prices_all.loc[:,'Cash']=pd.Series(1,index=prices_all.index)

            #dfTrades is initialized with the prices dataframe, but will eventually be filled with trades.
            dfTrades = prices_all.copy()
            dfTrades.ix[:] = 0

            for index, row in ordersDataFrame.iterrows():
                if row['Order'] == "BUY":
                    # print 'float(dfTrades.loc[pd.Timestamp.date(index), row[Symbol]])+ float(row[Shares])'
                    # print float(dfTrades.loc[pd.Timestamp.date(index), row['Symbol']])+ float(row['Shares'])
                    # print 'float(dfTrades.loc[pd.Timestamp.date(index), Cash]) '
                    # print float(dfTrades.loc[pd.Timestamp.date(index), 'Cash'])
                    # print 'float(row[Shares] *-1 * prices_all.loc[pd.Timestamp.date(index),row[Symbol]])'
                    # print float(row['Shares'] *-1 * prices_all.loc[pd.Timestamp.date(index),row['Symbol']])
                    # print 'row[Symbol]'
                    # print row['Symbol']
                    # print 'print row[Shares]'
                    # print float(row['Shares'])
                    # print 'float(dfTrades.loc[pd.Timestamp.date(index), row[Symbol]])'
                    # print float(dfTrades.loc[pd.Timestamp.date(index), row['Symbol']])
                    #dfTrades[date,symbol] = (whatever is in dfTrades, which should be 0 ) + sharesOrdered
                    if (float(dfTrades.loc[pd.Timestamp.date(index), row['Symbol']]))!= 0:
                        print ' tell me'
                    dfTrades.loc[pd.Timestamp.date(index), row['Symbol']] = float(row['Shares'])
                    dfTrades.loc[pd.Timestamp.date(index), 'Cash'] = float(dfTrades.loc[pd.Timestamp.date(index), 'Cash']) + float(row['Shares'] *-1 * prices_all.loc[pd.Timestamp.date(index),row['Symbol']])

                elif row['Order']== "SELL":
                    dfTrades.loc[pd.Timestamp.date(index), row['Symbol']] =float(dfTrades.loc[pd.Timestamp.date(index), row['Symbol']])+ (-1 * float(row['Shares']))
                    dfTrades.loc[pd.Timestamp.date(index), 'Cash'] = float(dfTrades.loc[pd.Timestamp.date(index), 'Cash']) + float(row['Shares'])  * float(prices_all.loc[pd.Timestamp.date(index),row['Symbol']])
            #After the above for loop, dfTrades now contains holdings on everyday of the time period.
            dfHoldings = dfTrades.copy()

            #sets everything in dfHoldings to 0
            dfHoldings.ix[:] = 0

            #first row of dfHOldings = any shares bought on day 1. cash = start value - change in cash on day 1
            #all other rows of dfHoldings = shares from current day-1 + any change in current day
            for i in range (dfHoldings.shape[1]):
                dfHoldings.ix[0,i] = dfTrades.ix[0,i]

            dfHoldings.ix[0, dfHoldings.shape[1]-1] = start_val+float(dfTrades.ix[0,dfTrades.shape[1]-1])

            dfHoldings[:] = dfTrades.cumsum()

            dfHoldings.ix[:,-1]= dfHoldings.ix[:,-1] + 1000000

            dfValues = dfHoldings * prices_all

            exceededDate, exceedsLeverage = self.getLeverageInfo(dfValues, exceededDate, exceedsLeverage)

            if exceedsLeverage == True:
                continue
            else:
                break
        portfolio_val = dfValues.sum(axis=1)

        return portfolio_val

    def getLeverageInfo(self, dfValues, exceededDate, exceedsLeverage):
        leverage = dfValues.copy()
        absoluteLeverage = dfValues.copy()
        allColumnsExceptCash = list(dfValues)
        allColumnsExceptCash.remove('Cash')
        absoluteLeverage.ix[:] = np.abs(absoluteLeverage.ix[:])
        leverage['leverage'] = absoluteLeverage[allColumnsExceptCash].sum(axis=1) / leverage.sum(axis=1)
        exceededLeverage = leverage[np.abs(leverage.leverage) > 2.0]
        if exceededLeverage.shape[0] > 0:
            exceededDate = exceededLeverage.index[0]
            exceedsLeverage = True
        else:
            exceedsLeverage = False
        return exceededDate, exceedsLeverage

    #Converts numbered indices to dates within a trading period.
    #Takes a dataframe of trades.  So each row is a trade, with column 0 indicating
    #the trade occurs on the ith day within the trading period, column 1 indicating the symbol, column
    #2 indicating the action, and column 3 indicating the quantity of the action.
    #So if a row is [5, SH, BUY, -200], it means on the fifth day of the trading period, sell 200 shares of
    #SH.
    #TODO refactor this method to return a series of dates rather than a dataframe.
    def convertIndicesToDates(self, ordersDF, prices_all):

        originalDF = ordersDF.copy()

        ordersDF = pd.DataFrame(index=prices_all.index, data=ordersDF, columns=ordersDF.columns)
        ordersDF = ordersDF.drop(ordersDF.columns[[0]], axis=1)
        ordersDF.columns = ['Symbol', 'Order', 'Shares']

        for i in range(0, originalDF.shape[0]):
            if i == 0:
                pass
            else:
                #once originalDF.ix[i, 0] == 0, that means there are no more orders.
                if originalDF.ix[i, 0] == 0:
                    break
            ordersDF.ix[int(originalDF.ix[i, 0]), 0] = originalDF.ix[i, 1]
            ordersDF.ix[int(originalDF.ix[i, 0]), 1] = originalDF.ix[i, 2]
            ordersDF.ix[int(originalDF.ix[i, 0]), 2] = int(originalDF.ix[i, 3])

        ordersDF=ordersDF.dropna(axis=0)

        return ordersDF

    def showChart(self, portfolio,dates,symbol,startingPortfolioValue):
        print portfolio

        portvals=self.computeDailyPortfolioValues(portfolio,dates,symbol,start_val = 1)

        stock = get_data([symbol],dates,addSPY=False)
        stock = stock.dropna()

        stock = (stock / stock.ix[0,:])*startingPortfolioValue

        plt.plot(stock.index,portvals,label = "Portvals")
        plt.plot(stock.index,stock.iloc[:,0],label = symbol)
        plt.legend(loc='upper left')

        plt.show()