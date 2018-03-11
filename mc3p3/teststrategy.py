"""
Test a Strategy Learner.  (c) 2016 Tucker Balch
"""

import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl

def test_code(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "GOOG"
    stdate =dt.datetime(2008,1,1)
    enddate =dt.datetime(2008,6,15) # just a few days for "shake out"

    # train the learner
    learner.addEvidence(symbol = sym, startDate= stdate, \
        endDate= enddate, sv = 10000)

    # set parameters for testing
    sym = "SH"
    stdate =dt.datetime(2009,1,1)
    enddate =dt.datetime(2009,5,15)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # test the learner and put results into a dataframe of trades to make per date
    portfolio = learner.testPolicy(symbol = sym, startDate= stdate, \
        endDate= enddate,isPortfolio=True)

    learner.showChart(portfolio,dates,sym,1000000)

    #learner.compute_portvals(df_trades,dates,"IBM")

if __name__=="__main__":
    test_code(verb = True)
