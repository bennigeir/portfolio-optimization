# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:33:04 2020
https://algotrading101.com/learn/yahoo-finance-api-guide/
@author: Gudmundur
"""

#%%
from yahoo_fin.stock_info import get_data
import pandas as pd
import numpy as np
#%%
def get_returns(tickerlist, start_date, end_date):
    """
    Function that generates daily returns for multiple tickers.
    Returns are not annualized.
    If there is only one ticker, it is much wiser to use the get_data function.
    
    Parameters
    ----------
    ticker : list of strings
        Strings are stock tickers.
    start_date : date string
        On the form dd/mm/yyyy. Indicates the initial observation date.
    end_date : TYPE
        On the form dd/mm/yyyy. Indicates the last observation date.

    Returns
    -------
    returns : DataFrame
        Daily returns for each stock ticker.

    """

    historical_datas = {}
    returns = pd.DataFrame()
    time_series_close = pd.DataFrame()
    
    for ticker in tickerlist:
        # print(ticker)
        historical_datas[ticker] = get_data(ticker, start_date, end_date)
        returns[ticker] = np.log(historical_datas[ticker]['adjclose']/
                                   historical_datas[ticker]['adjclose'].shift())
        time_series_close[ticker] = historical_datas[ticker]['adjclose']
    return returns, time_series_close

#%%
#Single financial intrument
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    """
    Calculate the annualized returns and risk for a portfolio with a set of weights

    Parameters
    ----------
    weights : numpy array
        Array with the weights for each instrument.
    mean_returns : DataFrame
        Dataframe with the returns for each intrument.
    cov_matrix : DataFrame
        The covariance matrix for the returns.

    Returns
    -------
    std : float
        Annualized standard deviation for the portfolio.
    returns : float
        Annualized return for the portfolio.

    """
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns



def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    """
    Monte Carlo simulation of portfolio generation without allocation. 
    Substitutes QP solvers.

    Parameters
    ----------
    num_portfolios : int
        Number of simulations.
    mean_returns : DataFrame
        Dataframe with the returns for each intrument.
    cov_matrix : DataFrame
        The covariance matrix for the returns.
    risk_free_rate : float
        The risk free rate at the given time (i.e. policy rates).

    Returns
    -------
    results : numpy array
        3d array that contains the annualized risk and return and the sharpe 
        ratio for the simulated portfolio.
    weights_record : numpy array
        Array that stores the weights for the simulated portfolios.

    """
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
#%%

#%% Construct a rebalancing method
"""
1. get the closing prices
2. calculate new optimal portfolio (monte carlo or theoretical)
3. evaluate new weights metrics
4. design a cost function to determine if the new weights are better
5. if changed calculate how much it would cost

cost function could evaluate if the cost of changing the porfolio will pay

"""

#%%


def pen_random_portfolios(num_simul, w1_ret, w2_ret, w3_ret, w1_allo, w2_allo, w3_allo, cov_mat, rf):
    results = np.zeros((3,num_simul))
    weights_record = []
    mean_returns = np.concatenate((w1_ret.mean(), w2_ret.mean(), w3_ret.mean()))
    cov_matrix = cov_mat
    
    for i in range(num_simul):
        w1 = np.random.random(len(w1_ret.columns))
        w1 /= np.sum(w1)
        w1 = w1 * w1_allo
        
        w2 = np.random.random(len(w2_ret.columns))
        w2 /= np.sum(w2) 
        w2 = w2 * w2_allo
        
        w3 = np.random.random(len(w3_ret.columns))
        w3 /= np.sum(w3)
        w3 = w3 * w3_allo
        
        con_weights = np.concatenate([w1,w2,w3])
        
        weights_record.append(con_weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(con_weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - rf) / portfolio_std_dev
    return results, weights_record

def pen_random_portfolios2(num_simul, w1_ret, w1_allo, cov_mat, rf):
    results = np.zeros((3,num_simul))
    weights_record = []
    mean_returns = w1_ret.mean()
    cov_matrix = cov_mat
    
    for i in range(num_simul):
        w1 = np.random.random(len(w1_ret.columns))
        w1 /= np.sum(w1)
        w1 = w1 * w1_allo
        
        con_weights = np.concatenate([w1])
        
        weights_record.append(con_weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(con_weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - rf) / portfolio_std_dev
    return results, weights_record

#%%



# returns.mean()*252 annualize log returns
# returns.std()*np.sqrt(252) annualize stedv
# 100*(1+returns_train['eik.ic']).cumprod()
# (100*(1+returns_train).cumprod()).plot()