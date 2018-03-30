__author__ = 'davidvinegar'
from learner import learner
import pandas as pd
import util
import matplotlib.pyplot as plt
import datetime as dt
import KNNLearner as knn

l = learner()
dates = pd.date_range('2007-12-31', '2009-12-31')
symbol = "IBM"

momentumDF = l.getMomentum(dates,symbol)
fiveDayPriceChange = l.getWeekPercentPriceChange(dates,symbol)
volatilityDF = l.getVolatility(dates,symbol)
bollingerBandDf = l.getBollingerBandVAlue(symbol,dates,volatilityDF)

stats = l.getStats(momentumDF,volatilityDF,bollingerBandDf)

bollingerBandDf = l.normalizeDataFrame(bollingerBandDf)
momentumDF = l.normalizeDataFrame(momentumDF)
volatilityDF = l.normalizeDataFrame(volatilityDF)

unalteredPrices = util.get_data([symbol],dates,addSPY=False).dropna()
fiveDayPriceChange, trainX, trainY,unalteredPrices = l.prepareTrainXandY(bollingerBandDf,
                                                                             fiveDayPriceChange, momentumDF,
                                                                             unalteredPrices, volatilityDF,symbol)

#Uncomment the LinRegl and comment the KNN Learner to use that instead of KNN
# learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
learner = knn.KNNLearner(2,verbose = True) # create a knn learner
learner.addEvidence(trainX, trainY) # train it


predictedYFromTraining = learner.query(
    trainX)  # get the predictions        sy = sknn.fit(trainX, trainY).predict(testX)
yPredictedDF = pd.DataFrame(predictedYFromTraining, index=fiveDayPriceChange.index)
yPredTimesPriceDF = yPredictedDF.values * unalteredPrices
fiveDayPrices = fiveDayPriceChange.values * unalteredPrices
yPredTimesPriceDF.columns = ['Predicted Y']
fiveDayPrices.columns = ['Y Train']
symbols = [symbol]
unalteredPrices = util.get_data(symbols, dates, addSPY=False)
unalteredPrices = unalteredPrices.dropna()
normalizedDailyPrices = unalteredPrices / unalteredPrices.ix[0, :]

l.showChartYTrainYPred(fiveDayPrices, unalteredPrices, yPredTimesPriceDF)


#learner,data, stats,unalteredPrices = l.setUp(dates,symbol)
#


l.trade(yPredTimesPriceDF,unalteredPrices,symbol)
portvals = learner.compute_portvals()
portvals.columns = ["Portfolio"]

# portvals.title = "Portfolio"
stock = util.get_data([symbol],dates,addSPY=False)
stock = stock.dropna()
stock = (stock / stock.ix[0,:])*10000

ax = stock.plot(title = "Daily Portfolio Value", mark_right = False)
ax.set_xlabel("Date")
ax.set_ylabel("Normalized price")

portvals.plot(label = "Portfolio", ax=ax,color = 'r')
plt.show()
cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = l.assess_portfolio(portvals)

testDates=  pd.date_range('2009-12-31', '2011-12-31')

testData = l.setUpTestData(testDates,learner,stats,symbol)

unalteredTestPrices = util.get_data([symbol], testDates,addSPY=False).dropna()
unalteredTestPrices = unalteredTestPrices / unalteredTestPrices.ix[0,:]
l.trade(testData, unalteredTestPrices,symbol)

portvalsTest = learner.compute_portvals(endDate=dt.date(2011,12,31))

portvalsTest.columns = ['Out Sample Portfolio']

stock = util.get_data([symbol],testDates,addSPY=False)
stock = stock.dropna()

stock = (stock / stock.ix[0,:])*10000
axTest = stock.plot(title = "Daily Portfolio Value-Out Sample", mark_right = False)

portvalsTest.plot(label = 'Out Sample Portfolio',ax=axTest,color = 'r')

axTest.set_xlabel("Date")
axTest.set_ylabel("Normalized price")
plt.show()
cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = l.assess_portfolio(portvalsTest)