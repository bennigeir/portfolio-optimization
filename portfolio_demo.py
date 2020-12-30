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
df_omx_train = df_omx[(df_omx.Date >= '2016-01-01') & (df_omx.Date <= '2018-12-01')]

df_iv_train = df_iv.loc['2016-01-01':'2018-12-01']

df_stocks_train = df_stocks.loc['2016-01-01':'2018-12-01']

df_is_train = df_is.loc['2016-01-01':'2018-12-01']

# Hér væri sniðugt að taka út nan gildi og jafna út lengd á dataframeum
# Þ.e. sá minnsti DF stýrir hvaða dagar eru teknir inn
# Hér væru líka þau bréf valin sem ættu að taka í reikningin og flokka
# eins og skuldabréf, hlutabréf og forex.


print('GENERATE RETURNS AND COV MATRIX')
omx_returns_train = np.log(df_omx_train['OMXI10']/df_omx_train['OMXI10'].shift())

iv_returns_train = np.log(df_iv_train/df_iv_train.shift())

stocks_returns_train = np.log(df_stocks_train/df_stocks_train.shift())

is_return_trains = np.log(df_is_train/df_is_train.shift())


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

num_sims = 150000
w_iv = 0.2
w_stocks = 0.6
w_is = 0.2

rf = 0.03

results_rand, weights_rand = pen_random_portfolios(num_sims, iv_returns_train, 
                                                   stocks_returns_train, is_return_trains,
                                                   w_iv, w_stocks, w_is, cov_matrix, rf)

#%%
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(results_rand[0], results_rand[1], c=results_rand[2], cmap=cm)
plt.colorbar(sc)
plt.xlabel('RISK')
plt.ylabel('EXPECTED RETURN')
plt.title('MONTE CARLO PORTFOLIO SIMULATIONS')



max_sharpe_idx = np.argmax(results_rand[2])
sdp, rp = results_rand[0,max_sharpe_idx], results_rand[1,max_sharpe_idx]

plt.scatter(sdp,rp, s=40, marker='x')

max_sharpe_allocation = pd.DataFrame(weights_rand[max_sharpe_idx],index=cov_matrix.columns,columns=['allocation'])
# max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
max_sharpe_allocation = max_sharpe_allocation.T
plt.show()
print ("-"*80)
print ("Maximum Sharpe Ratio Portfolio Allocation\n")
print ("Annualised Return:", round(rp,2))
print ("Annualised Volatility:", round(sdp,2))
print ("\n")
print (max_sharpe_allocation)

#%% BACKTESTING

"""
Ef þetta væri real time væri þetta skref aðeins öðruvísi

"""
print('GENERATING TEST DATA')
df_omx_test = df_omx[(df_omx.Date >= '2018-12-01') & (df_omx.Date <= '2020-12-01')]
df_iv_test = df_iv.loc['2018-12-01':'2020-12-01']
df_stocks_test = df_stocks.loc['2018-12-01':'2020-12-01']
df_is_test = df_is.loc['2018-12-01':'2020-12-01']


omx_returns_test = np.log(df_omx_test['OMXI10']/df_omx_test['OMXI10'].shift()).reset_index(drop=True)
iv_returns_test = np.log(df_iv_test/df_iv_test.shift())
stocks_returns_test= np.log(df_stocks_test/df_stocks_test.shift())
is_return_test = np.log(df_is_test/df_is_test.shift())

#Reseta indexinn þangað til ég finn úr því hvernig ég get sameinað þá og þeir séu jafn langir
#hugmynd að nota isin (iv_returns_test[iv_returns_test.index.isin(is_return_test.index)]) eitthvað sem svipar til þssa
test_returns = (pd.concat([iv_returns_test.reset_index(drop=True),
                           stocks_returns_test.reset_index(drop=True), 
                           is_return_test.reset_index(drop=True)], axis=1))


port_returns = test_returns.dot(max_sharpe_allocation.T)

plt.plot(port_returns.cumsum())
plt.plot(omx_returns_test.cumsum())
plt.ylabel('RETURN')
plt.xlabel('TIME')
plt.title('OPT PORTFOLIO VS OMXI10')
plt.legend(('Anal Bois', 'shitty market'),
            loc='upper left')


