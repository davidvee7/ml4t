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

    def addEvidence(self,symbol = "IBM", sd=dt.datetime(2007,12,31), ed=dt.datetime(2009,12,31), sv = 10000):
        dates = pd.date_range(sd, ed)
        allDf = self.setUp(dates,symbol)

        # print "initial size post set up. maybe set up mutates it"
        # print allDf.shape

        allDf = allDf[:,:]

        data = self.discretize(allDf)
        # print "proper discretize"
        # print data
        # print "discretized"
        # print data
        # benchmark = allDf[:,-1].sum() - allDf.shape[0]
        # print "benchmark is "
        # print benchmark
        # self.QLearner.querysetstate()
        shape = allDf.shape
        # print type(data)

        data = data.loc[1:,:]
        # ones = np.zeros((shape[0],1))
        # dfTrades = pd.DataFrame(ones)


        for iteration in range(0,200):
            if time.time() >=self.endtime:
                break
            # dfTrades.iloc[0] = 0
            # dfTrades.iloc[1] = 0
            # dfTrades.iloc[2] = 0

            steps=2
            robopos = 0

            # print data

            datum = data.iloc[4,:] #convert the location to a state
            state = self.makeIntoState(datum,robopos)
            action = self.QLearner.querysetstate(state) #set the state and get first action
            cumulativeRewards = 0
            r=0
            for i in range (3,shape[0]-1):
                # print " action"
                # print action
                # print robopos
                newpos = self.movebot(robopos,action)

                #move to new location according to action and then get a new action

                if i >0:# shape[0]-1:
                    if newpos == 1:
                        r = allDf[i,-1]-1
                        # print "bout to add below r to cumulativerewards"
                        # print r

                        cumulativeRewards+=r
                    elif newpos == 2:
                        r = (allDf[i,-1]-1)*-1
                        cumulativeRewards+=r

                    else:
                        # print "dis what robopos equals after rejection"
                        # print robopos
                        r = 0
                else:
                    state= self.makeIntoState(datum,newpos)
                    # dfTrades.iloc[i] = 0
                    robopos = newpos
                    steps+=1
                    continue

                datum = data.iloc[i,:]
                state = self.makeIntoState(datum,newpos)
                prevAction = action
                action = self.QLearner.query(state,r)

                robopos = action
                steps += 1
        # print iteration, "," , cumulativeRewards
            # print dfTrades

    def makeIntoState(self,data,currentPos):
        # print "problem causers"
        # print currentPos
        # print data
        dataStr = str(int(currentPos))
        dataStr+=str(int(data.iloc[0]))


        dataStr+=str(int(data.iloc[1]))
        dataStr+=str(int(data.iloc[2]))


        # dataStr = str(int(currentPos))+ str(int(data.iloc[0]))+ str(int(data.iloc[1])) + str(int(data.iloc[2]))
        # print "franken string"
        # print dataStr
        dataStr = int(dataStr)
        dataStr = dataStr% 250

        return dataStr

    def testPolicy(self,symbol = "IBM", sd=dt.datetime(2007,12,31), ed=dt.datetime(2009,12,31), sv = 10000):

        self.QLearner.rar = 0
        dates = pd.date_range(sd, ed)
        allDf = self.setUp(dates,symbol)

        allDf = allDf[:,:]


        data = self.discretizeTest(allDf)
        # print "test discretized"
        # print data
        shape = allDf.shape
        # print type(data)

        # data = data.loc[1:,:]
        zeros = np.zeros((shape[0],1))
        dfTrades = pd.DataFrame(zeros)
        portfolioND = np.zeros((shape[0],4))
        portfolio = pd.DataFrame(portfolioND)

        steps = 0
        robopos = 0
        # print "data below"
        # print data
        datum = data.iloc[0,:] #convert the location to a state
        # print "datum and robopos"
        # print datum
        # print robopos
        # print "robopos above"
        state = self.makeIntoState(datum,robopos)
        action = self.QLearner.querysetstate(state) #set the state and get first action
        cumulativeRewards = 0
        r=0

        holdings = 0
        portfolioCount=0
        for i in range (0,shape[0]-1):

            newpos = self.movebot(robopos,action)
            prevAction = robopos
            if (newpos ==1 or newpos == 2) and (prevAction==0):
                holding = 1
            if (newpos==0) and (prevAction==1 or prevAction==2):
                holding = 0
            if i >0:# shape[0]-1:
                if newpos == 1:
                    r = allDf[i,-1]-1
                    cumulativeRewards+=r
                elif newpos == 2:
                    r = (allDf[i,-1]-1)*-1
                    cumulativeRewards+=r

                else:
                    r = 0
            # else:
            #     state = self.makeIntoState(datum,newpos)
            #     dfTrades.iloc[i] = 0
            #     robopos=newpos
            #     steps+=1
            #     # print "prevaction and actoin " + str(prevAction),str(action)
            #     continue
            datum = data.iloc[i,:]
            state = self.makeIntoState(datum,newpos)
            action = self.QLearner.querysetstate(state)
            # print dfTrades.shape
# 0 2  2,1 1,1   1,1   1,2   2,0,  #0,1 1,0 0,1,10, 02, 2,0 0,1 1,2 2,2 2,1 1,0

            # print "i == " + str(i) + " and below robopos, action"
            # print robopos, action

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
                portfolio.loc[portfolioCount,1] = "IBM"
                portfolio.loc[portfolioCount,2] = "SELL"
                portfolio.loc[portfolioCount,3] = -200
                portfolioCount+=1
            elif robopos == 1 and action == 0:
                dfTrades.iloc[i] = -100
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = "IBM"
                portfolio.loc[portfolioCount,2] = "SELL"
                portfolio.loc[portfolioCount,3] = -100
                portfolioCount+=1

            elif robopos == 2 and action == 1:
                dfTrades.iloc[i] = 200
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = "IBM"
                portfolio.loc[portfolioCount,2] = "BUY"
                portfolio.loc[portfolioCount,3] = 200
                portfolioCount+=1

            elif robopos == 2 and action == 0:
                dfTrades.iloc[i] = 100
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = "IBM"
                portfolio.loc[portfolioCount,2] = "BUY"
                portfolio.loc[portfolioCount,3] = 100
                portfolioCount+=1

            elif robopos == 0 and action == 1:
                dfTrades.iloc[i] = 100
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = "IBM"
                portfolio.loc[portfolioCount,2] = "BUY"
                portfolio.loc[portfolioCount,3] = 100
                portfolioCount+=1

            elif robopos == 0 and action == 2:
                dfTrades.iloc[i] = -100
                portfolio.loc[portfolioCount,0] = i
                portfolio.loc[portfolioCount,1] = "IBM"
                portfolio.loc[portfolioCount,2] = "SELL"
                portfolio.loc[portfolioCount,3] = -100
                portfolioCount+=1

            # print "prevactiona nd action"
            # print prevAction,action
            # dfTrades.iloc[i] = int(str(action)[0])
            robopos = action
            steps += 1
        # print "Cum Rewards" , cumulativeRewards
        # print portfolio
        # print "dftrades"
        # print dfTrades
        # portvals=self.compute_portvals(portfolio,dates,symbol,start_val = 1)
        # portvals.columns = ['Out Sample Portfolio']
        # print "columns of portval"
        # print portvalsTest.columns
        # stock = get_data(["IBM"],dates,addSPY=False)
        # stock = stock.dropna()
        #
        # stock = (stock / stock.ix[0,:])*1000000
        # axTest = stock.plot(title = "Daily Portfolio Value-In Sample", mark_right = False)
        #
        # portvals.plot(label = 'Out Sample Portfolio',ax=axTest,color = 'r')
        #
        # axTest.set_xlabel("Date")
        # axTest.set_ylabel("Normalized price")
        # plt.show()
        # print "dftrades final shape"
        # print dfTrades.shape
        return dfTrades

    def discretizeTest(self,data):
        # print
        # print self.thresholds0[1]

        # dataDF= pd.DataFrame(data)
        data = data[1:,:]
        data = pd.DataFrame(data)
        # print "data problem"
        # print data

        ndData = data.as_matrix()

        nd1 = data.ix[:,0]
        nd2 = data.ix[:,1]
        nd3= data.ix[:,2]
        # print "nd1 below"
        # print nd1

        bins1 = self.thresholds0[1]
        bins2= self.thresholds1[1]
        bins3=self.thresholds2[1]

        # print "the bins"
        # print bins1
        # print "bins 2"
        # print bins2
        # print bins3
        #
        # print "are ther erounding errors"
        # print nd1
        cut1=np.digitize(nd1,bins1) -1
        cut2= np.digitize(nd2,bins2) -1
        cut3= np.digitize(nd3,bins3) -1

        cut1[cut1>9] = 9
        cut1[cut1<0] = 0
        cut2[cut2>9] = 9
        cut2[cut2<0] = 0
        cut3[cut3>9] = 9
        cut3[cut3<0] = 0
        # cut1 = pd.cut(nd1,bins1)
        # print "cut1"
        # print cut1
        df1 = pd.DataFrame(cut1)
        df2 = pd.DataFrame(cut2)
        df3= pd.DataFrame(cut3)
        # cut1.categories=[0,1,2,3,4,5,6,7,8,9]
        # print "modified cut1"
        # print cut1
        df = pd.concat([df1,df2],axis=1)
        df = pd.concat([df,df3],axis=1)
        # df[:,:] = df-1
        # print "df here"
        # print df
        return df

    def discretize(self,data):
        dataDF= pd.DataFrame(data)
        dataDF= dataDF.ix[1:,:]
        # print dataDF
        self.thresholds0 = pd.qcut(dataDF.ix[:,0],10,labels=False,retbins=True)
        self.thresholds1 = pd.qcut(dataDF.ix[:,1],10,labels=False,retbins=True)
        self.thresholds2 = pd.qcut(dataDF.ix[:,2],10,labels=False,retbins=True)

        # "tresholds 1!!"
        # print self.thresholds0[1]

        thresholds0DF= pd.DataFrame(self.thresholds0[0])
        thresholds1DF= pd.DataFrame(self.thresholds1[0])
        thresholds2DF= pd.DataFrame(self.thresholds2[0])


        allDF = thresholds0DF.join(thresholds1DF, how ='inner')
        allDF = allDF.join(thresholds2DF, how = "inner")
        # allDF = allDF.ix[1:,:]
        # print allDF

        return allDF
    #Gets momentum, volatility, spy momentum, and bb values.  puts them into a single ndarray.  also puts averages and
    # standard deviations into a stats list variable for use later in normalization
    def setUp(self,dates,symbols):
        # print symbols
        # print dates
        momentumDF, actual5DayChange, unalteredPrices = self.getMomentum(dates,symbols)
        # print "momentum df"
        # print momentumDF.shape
        spyMomentum = self.getSPYMomentum(dates)
        stdDF=self.getVolatility(dates,symbols)

        # print stdDF.shape

        SMA= pd.rolling_mean(unalteredPrices, window = 3)
        SMA= SMA.dropna()
        bb_value = (unalteredPrices-SMA)/(2*stdDF)

        # print bb_value.shape
        stats = []


        # momentumDF = momentumDF[1:-1]
        # stdDF = stdDF[1:-1]
        # actual5DayChange = actual5DayChange[1:-1]
        # bb_value = bb_value[1:-1]
        unalteredPrices= unalteredPrices/unalteredPrices.ix[0,:]
        # unalteredPrices = unalteredPrices[1:-1]

        actual5DayChange = actual5DayChange+1

        allDF = np.ones((momentumDF.shape[0],4))
        allDF[:,0]= momentumDF[symbols]
        allDF[:,1]= stdDF[symbols]
        allDF[:,2]= bb_value[symbols]
        allDF[:,3]= actual5DayChange[symbols]

        trainX = allDF[:,0:-1]
        trainY = allDF[:,-1]

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
        # dates = pd.date_range('2007-12-31', '2009-12-31')
        symbols = [symbols]
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

    def movebot(self,oldpos,a):
        #a == 0 means hold, 1 means buy

        if a ==oldpos:
            pass
        else:
            oldpos = a
        newpos = oldpos
        return newpos

    def compute_portvals(self,ordersDF,dateRange,symbol,start_val = 1):
        # this is the function the autograder will call to test your code
        # TODO: Your code here
        # orders_all = get_data(syms, dates)  # automatically adds SPY
        exceedsLeverage = True
        exceededDate = None
        # ordersDF= pd.read_csv(orders_file, index_col = "Date", parse_dates = True, usecols = ['Date', 'Symbol','Order','Shares'])

        originalDF = ordersDF.copy()
        prices_all = get_data([symbol], dateRange)

        ordersDF = pd.DataFrame(index=prices_all.index,data=ordersDF,columns = ordersDF.columns)
        # print "ordersDf before mutation"
        # print ordersDF
        ordersDF=ordersDF.drop(ordersDF.columns[[0]],axis=1)

        ordersDF.columns = ['Symbol','Order','Shares']
        # print "newdf"
        # print ordersDF

        # print "original df before"
        # print originalDF
        for i in range(0,originalDF.shape[0]):
            if i== 0:
                pass
            else:
                if originalDF.ix[i,0]==0:
                    break
            ordersDF.ix[originalDF.ix[i,0],0] = originalDF.ix[i,1]
            ordersDF.ix[originalDF.ix[i,0],1] = originalDF.ix[i,2]
            ordersDF.ix[originalDF.ix[i,0],2] = originalDF.ix[i,3]

        # print "should be good"
        ordersDF=ordersDF.dropna(axis=0)
        # print ordersDF


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

            sd = prices_all.index.min()
            ed = prices_all.index.max()
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

            # print " ordersDF"
            # print ordersDF



            for index, row in ordersDF.iterrows():
                # print "shitty index"
                # print pd.Timestamp.date(index)
                if row['Order'] == "BUY":
                    # print "here's dftrades"
                    # print dfTrades
                    dfTrades.loc[pd.Timestamp.date(index), row['Symbol']] = float(dfTrades.loc[pd.Timestamp.date(index), row['Symbol']])+ float(row['Shares'])
                    dfTrades.loc[pd.Timestamp.date(index), 'Cash'] = float(dfTrades.loc[pd.Timestamp.date(index), 'Cash']) + float(row['Shares'] *-1 * dfPrices.loc[pd.Timestamp.date(index),row['Symbol']])

                elif row['Order']== "SELL":
                    # print "here's dftrades"
                    # print dfTrades
                    dfTrades.loc[pd.Timestamp.date(index), row['Symbol']] =float(dfTrades.loc[pd.Timestamp.date(index), row['Symbol']])+ (-1 * float(row['Shares']))
                    dfTrades.loc[pd.Timestamp.date(index), 'Cash'] = float(dfTrades.loc[pd.Timestamp.date(index), 'Cash']) + float(row['Shares'])  * float(dfPrices.loc[pd.Timestamp.date(index),row['Symbol']])

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
            dfHoldings.ix[:,-1]= dfHoldings.ix[:,-1] + 1000000

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
        # print "portfolio_val"
        # print portfolio_val
        # print portfolio_val
        return portfolio_val

learner = sl.StrategyLearner(verbose = False) # constructor
learner.addEvidence(symbol = "IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv = 10000) # training step
df_trades = learner.testPolicy(symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv = 10000) # testing step
