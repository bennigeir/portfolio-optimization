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
from datetime import timedelta 

#%% READ DATA
print('READING DATA')

df_omx = get_omx()

df_iv = get_iv()

df_stocks = get_stocks()

df_is = get_is()

print ('DATA READING FINISHED')

#%% SELECT INSTRUMENTS

# Hér væri sniðugt að taka út nan gildi og jafna út lengd á dataframeum
# Þ.e. sá minnsti DF stýrir hvaða dagar eru teknir inn
# Hér væru líka þau bréf valin sem ættu að taka í reikningin og flokka
# eins og skuldabréf, hlutabréf og forex.
# Gæti notað clustering til að velja bréfin og taka bréfið sem er með max shrape
# value í hverjum cluster

df_iv_filt = df_iv[['ÍV Erlent hlutabréfasafn - Runugildi',
                          'ÍV Alþjóðlegur hlutabréfasjóður - Runugildi']]

df_is_filt = df_is[['IS-RIKISSKULDABREF-LONG',
                    'IS-SERTRYGGD-SKULDABREF']]

df_stocks_filt = df_stocks[['MAREL.IC', 'FESTI.IC',
                            'VIS.IC', 'ARION.IC',
                            'ICEAIR.IC']]


#%% DATA PREPERATION
print('FILTER DATA')
df_omx_train = df_omx[(df_omx.Date >= '2016-01-01') & (df_omx.Date <= '2018-12-01')]

df_iv_train = df_iv_filt.loc['2016-01-01':'2018-12-01']

df_stocks_train = df_stocks_filt.loc['2016-01-01':'2018-12-01']

df_is_train = df_is_filt.loc['2016-01-01':'2018-12-01']

df_iv_train.index = pd.to_datetime(df_iv_train.index)
df_stocks_train.index = pd.to_datetime(df_stocks_train.index)
df_is_train.index = pd.to_datetime(df_is_train.index)

df_value = pd.merge(df_iv_train, df_stocks_train, on='Date', how='outer')
df_value = pd.merge(df_value, df_is_train, on='Date', how='outer')
df_value = df_value.sort_index()


print('GENERATE RETURNS AND COV MATRIX')
omx_returns_train = np.log(df_omx_train['OMXI10']/df_omx_train['OMXI10'].shift())

train_returns = np.log(df_value/df_value.shift())

cov_matrix = train_returns.cov()


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

iv_returns_train = train_returns[['ÍV Erlent hlutabréfasafn - Runugildi',
                          'ÍV Alþjóðlegur hlutabréfasjóður - Runugildi']]

stocks_returns_train = train_returns[['MAREL.IC', 'FESTI.IC',
                                             'VIS.IC', 'ARION.IC',
                                             'ICEAIR.IC']]

is_return_trains = train_returns[['IS-RIKISSKULDABREF-LONG',
                                     'IS-SERTRYGGD-SKULDABREF']]

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
sharpe_value = results_rand[2,max_sharpe_idx]
sdp, rp = results_rand[0,max_sharpe_idx], results_rand[1,max_sharpe_idx]

plt.scatter(sdp,rp, s=400, marker='x', c='orange')

max_sharpe_allocation = pd.DataFrame(weights_rand[max_sharpe_idx],index=cov_matrix.columns,columns=['allocation'])
max_sharpe_allocation = max_sharpe_allocation.T
plt.show()

print ("-"*80)
print ("Maximum Sharpe Ratio Portfolio Allocation\n")
print ("Annualised Return:", round(rp,2))
print ("Annualised Volatility:", round(sdp,2))
print ("\n")
print (max_sharpe_allocation)

#%% BACKTESTING

print('GENERATING TEST DATA')
df_omx_test = df_omx[(df_omx.Date >= '2018-12-01') & (df_omx.Date <= '2020-12-01')]

df_iv_test = df_iv_filt.loc['2018-12-01':'2020-12-01']
df_stocks_test = df_stocks_filt.loc['2018-12-01':'2020-12-01']
df_is_test = df_is_filt.loc['2018-12-01':'2020-12-01']

omx_returns_test = df_omx_test.copy()
omx_returns_test['LOG'] = np.log(df_omx_test['OMXI10']/df_omx_test['OMXI10'].shift())
omx_returns_test = omx_returns_test.set_index(['Date'])

df_iv_test.index = pd.to_datetime(df_iv_test.index)
df_stocks_test.index = pd.to_datetime(df_stocks_test.index)
df_is_test.index = pd.to_datetime(df_is_test.index)

test_series = pd.merge(df_iv_test, df_stocks_test, on='Date', how='outer')
test_series = pd.merge(test_series, df_is_test, on='Date', how='outer')
test_series = test_series.sort_index()

test_returns = np.log(test_series/test_series.shift())

port_returns = test_returns.dot(max_sharpe_allocation.T)

plt.plot(port_returns.cumsum())
plt.plot(omx_returns_test['LOG'].cumsum())
plt.ylabel('RETURN')
plt.xlabel('TIME')
plt.title('OPT PORTFOLIO VS OMXI10')
plt.legend(('Anal Bois', 'shitty market'),
            loc='upper left')

#%% REBALANCING

df_iv_cols = df_iv_filt.columns
df_stock_cols = df_stocks_filt.columns
df_is_cols = df_is_filt.columns

shares_df = pd.DataFrame()
portfolio_df = pd.DataFrame()

initial_day = test_series.index[0]
initial_investment = 1e6
initial_portfolio = max_sharpe_allocation
initial_sharpe = sharpe_value
initial_std = sdp
initial_rp = rp

initial_fund_allocation = initial_investment*initial_portfolio
initial_nbr_shares = initial_fund_allocation/test_series.loc[initial_day] #nbr of shares

initial_list = [[initial_day, initial_investment, initial_sharpe]]
store_df = pd.DataFrame(initial_list, columns=['DATE','FUNDS','SHARPE'])


portfolio_df = portfolio_df.append(initial_portfolio)
portfolio_df['DATE'] = initial_day

shares_df = shares_df.append(initial_nbr_shares)
shares_df['DATE'] = initial_day

dates = port_returns.index
rollback_year = 2
rollback = timedelta(days=rollback_year*365)

rebalance_freq = 30
rebalance_threshold = 0.05

# temp_day = dates[30]



# i%30

#%%
print('EVALUATING THE PORTFOLIO')
for i in range(1,len(dates)):
    rebalance_bool = False
    
    if i%30==0:
        #Check the value of the portfolio
        temp_day = dates[i]
        # print(temp_day)
        temp_value = test_series.loc[temp_day]*shares_df.drop('DATE',axis=1).iloc[-1]
        
        temp_iv_w = temp_value[df_iv_cols].sum()/temp_value.sum()
        temp_stocks_w = temp_value[df_stock_cols].sum()/temp_value.sum()
        temp_is_w = temp_value[df_is_cols].sum()/temp_value.sum()
        
        #Check if the initial allocation is skewed
        iv_w_bool = abs(temp_iv_w-w_iv)>rebalance_threshold
        stocks_w_bool = abs(temp_stocks_w-w_stocks)>rebalance_threshold
        is_w_bool = abs(temp_is_w-w_is)>rebalance_threshold
        
        if (iv_w_bool+stocks_w_bool+is_w_bool)>0:
            print('INITIAL WEIGHTS SKEWED, REBALANCE NEEDED')
            rebalance_bool = True

        df_iv_filt.index = pd.to_datetime(df_iv_filt.index)
        df_stocks_filt.index = pd.to_datetime(df_stocks_filt.index)
        df_is_filt.index = pd.to_datetime(df_is_filt.index)
        
        df_iv_rollback = df_iv_filt.loc[temp_day-rollback:temp_day]
        df_stocks_rollback = df_stocks_filt.loc[temp_day-rollback:temp_day]
        df_is_rollback = df_is_filt.loc[temp_day-rollback:temp_day]
        
        
        df_rollback = pd.merge(df_iv_rollback, df_stocks_rollback, on='Date', how='outer')
        df_rollback = pd.merge(df_rollback, df_is_rollback, on='Date', how='outer')
        df_rollback = df_rollback.sort_index()
        
        
        rollback_returns = np.log(df_rollback/df_rollback.shift())
        
        roll_cov_matrix = rollback_returns.cov()


        num_sims = 1500
        w_iv = 0.2
        w_stocks = 0.6
        w_is = 0.2
        
        rf = 0.03
        
        iv_returns_roll = rollback_returns[['ÍV Erlent hlutabréfasafn - Runugildi',
                                  'ÍV Alþjóðlegur hlutabréfasjóður - Runugildi']]
        
        stocks_returns_roll = rollback_returns[['MAREL.IC', 'FESTI.IC',
                                                     'VIS.IC', 'ARION.IC',
                                                     'ICEAIR.IC']]
        
        is_return_roll = rollback_returns[['IS-RIKISSKULDABREF-LONG',
                                             'IS-SERTRYGGD-SKULDABREF']]
        
        results_roll, weights_roll = pen_random_portfolios(num_sims, iv_returns_roll, 
                                                           stocks_returns_roll, is_return_roll,
                                                           w_iv, w_stocks, w_is, roll_cov_matrix, rf)


        max_sharpe_idx_roll = np.argmax(results_roll[2])
        sharpe_value_roll = results_roll[2,max_sharpe_idx_roll]
        sdp_roll, rp_roll = results_roll[0,max_sharpe_idx_roll], results_roll[1,max_sharpe_idx_roll]
        
        if (sharpe_value_roll-store_df['SHARPE'].iloc[-1])>=0.05:
            print('SHARPE VALUE INCREASED, REBALANCE NEEDED')
            rebalance_bool = True
        
        if rebalance_bool:
            # print(i)
            print(temp_day)
            max_sharpe_allocation_roll = pd.DataFrame(weights_roll[max_sharpe_idx_roll],
                                                      index=roll_cov_matrix.columns,columns=['allocation'])
            
            max_sharpe_allocation_roll = max_sharpe_allocation_roll.T
            
            store_list = [[temp_day, temp_value.sum(), sharpe_value_roll]]
            store_df = store_df.append(store_list, ignore_index=True)
            
            fund_allocation = temp_value.sum()*max_sharpe_allocation_roll
            nbr_shares = fund_allocation/test_series.loc[temp_day]
            
            shares_df = shares_df.append(nbr_shares)
            shares_df['DATE'] = temp_day
            
            portfolio_df = portfolio_df.append(max_sharpe_allocation_roll)
            portfolio_df['DATE'] = 
            
            # As of now the DATE column is constant, need to fix so it updates and takes in every value not 
            # only the last value
            

















#tékka á ret og stdev með gömlum vigtum
#Rebalance getur átt sér stað ef asset allocation er orðið mjög skewed 
#eða ef áæltuð ávöxtunarkrafa hækkar mv.risk (i.e. sharpe)
#rebalance-a á mánaðar fresti