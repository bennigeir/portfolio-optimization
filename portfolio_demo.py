# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:49:15 2020
Demonstration for the optimal portfolio functions
@author: Gudmundur
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_omx, get_iv, get_stocks, get_is
from opt_functions import pen_random_portfolios

#%% READ DATA
print('READING DATA')

df_omx = get_omx()

df_iv = get_iv()

df_stocks = get_stocks()

df_is = get_is()

print ('DATA READING FINISHED')

#%% DATA PREPERATION
print('FILTER DATA')
df_omx = df_omx[(df_omx.Date >= '2016-01-01') & (df_omx.Date <= '2018-12-01')]

df_iv = df_iv.loc['2016-01-01':'2018-12-01']

df_stocks = df_stocks.loc['2016-01-01':'2018-12-01']

df_is = df_is.loc['2016-01-01':'2018-12-01']

# Hér væri sniðugt að taka út nan gildi og jafna út lengd á dataframeum
# Þ.e. sá minnsti DF stýrir hvaða dagar eru teknir inn
# Hér væru líka þau bréf valin sem ættu að taka í reikningin og flokka
# eins og skuldabréf, hlutabréf og forex.


print('GENERATE RETURNS AND COV MATRIX')
omx_returns_train = np.log(df_omx['OMXI10']/df_omx['OMXI10'].shift())

iv_returns_train = np.log(df_iv/df_iv.shift())

stocks_returns_train = np.log(df_stocks/df_stocks.shift())

is_return_trains = np.log(df_is/df_is.shift())


cov_matrix = (pd.concat([iv_returns_train.reset_index(drop=True),
                        stocks_returns_train.reset_index(drop=True), 
                        is_return_trains.reset_index(drop=True)], axis=1)).cov()


#%% PORTFOLIO GENERATION
"""
Optimization input

25000 simulations

Returns from IV, Stocks and IS

Define the allocation of funds in the portfolio:
    20% in funds from IV
    60% in stocks in NASDAQ IS
    20% in funds from IS
    
Risk free rate 3%

"""

num_sims = 10
w_iv = 0.2
w_stocks = 0.6
w_is = 0.2

rf = 0.03

results_rand, weights_rand = pen_random_portfolios(num_sims, iv_returns_train, 
                                                   stocks_returns_train, is_return_trains,
                                                   w_iv, w_stocks, w_is, cov_matrix, rf)

#%%

# max_sharpe_idx = np.argmax(results_rand[2])
# sdp, rp = results_rand[0,max_sharpe_idx], results_rand[1,max_sharpe_idx]
# max_sharpe_allocation = pd.DataFrame(weights_record_rand[max_sharpe_idx],index=con_col.columns,columns=['allocation'])
# # max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
# max_sharpe_allocation = max_sharpe_allocation.T

# print ("-"*80)
# print ("Maximum Sharpe Ratio Portfolio Allocation\n")
# print ("Annualised Return:", round(rp,2))
# print ("Annualised Volatility:", round(sdp,2))
# print ("\n")
# print (max_sharpe_allocation)


