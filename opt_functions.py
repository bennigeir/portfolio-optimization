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
ticker_list = ['arion.ic', 'brim.ic', 'eik.ic', 'eim.ic', 'festi.ic',
               'haga.ic', 'iceair.ic', 'icesea.ic', 'kvika.ic',
               'marel.ic', 'reginn.ic', 'reitir.ic','siminn.ic', 'sjova.ic',
                'skel.ic', 'syn.ic', 'tm.ic', 'vis.ic']

returns_train, hist_data_train = get_returns(ticker_list, "12/04/2017", "23/12/2019")

#%%
#Single financial intrument
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns



def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(returns_train.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
#%%
cov_matrix = returns_train.cov()
num_portfolios = 15000
risk_free_rate = 0.03

results_rand, weights_record_rand = random_portfolios(num_portfolios, returns_train.mean(), cov_matrix, risk_free_rate)

#%%

max_sharpe_idx = np.argmax(results_rand[2])
sdp, rp = results_rand[0,max_sharpe_idx], results_rand[1,max_sharpe_idx]
max_sharpe_allocation = pd.DataFrame(weights_record_rand[max_sharpe_idx],index=returns_train.columns,columns=['allocation'])
# max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
max_sharpe_allocation = max_sharpe_allocation.T

print ("-"*80)
print ("Maximum Sharpe Ratio Portfolio Allocation\n")
print ("Annualised Return:", round(rp,2))
print ("Annualised Volatility:", round(sdp,2))
print ("\n")
print (max_sharpe_allocation)
#%%
a = results_rand[0]
b = results_rand[1]



#%%
returns_test, close_test = get_returns(ticker_list, "24/12/2019", "23/12/2020")

#%% Construct a rebalancing method
"""
1. get the closing prices
2. calculate new optimal portfolio (monte carlo or theoretical)
3. evaluate new weights metrics
4. design a cost function to determine if the new weights are better
5. if changed calculate how much it would cost

cost function could evaluate if the cost of changing the porfolio will pay

"""
returns_train, hist_data_train = get_returns(ticker_list, "13/04/2017", "27/12/2019")



#%% Construct penalized optimal portfolio
"""
One should have the option of choosing the distribution of instruments i.e. 30% stocks, 30% bonds, 40% currency

"""
funds = pd.read_excel('is_funds_28122020.xls')

#%%
ticker_list = ['brim.ic', 'eik.ic', 'eim.ic', 'festi.ic',
               'haga.ic', 'iceair.ic', 'icesea.ic', 'kvika.ic',
               'marel.ic', 'reginn.ic', 'reitir.ic','siminn.ic', 'sjova.ic',
                'skel.ic', 'syn.ic', 'tm.ic', 'vis.ic']

returns_train, hist_data_train = get_returns(ticker_list, "12/04/2017", "29/01/2020")

#%%
fx_list = ['ISK=X', 'GBPISK=X', 'EURISK=X', 'JPYISK=X', 'CHFISK=X',
               'DKKISK=X', 'NOKISK=X']

fx_returns_train, fx_hist_data_train = get_returns(fx_list, "12/04/2017", "23/12/2019")

#%%
#Need to find a way to adress national holidays in iceland, for this time I just add some dates so the number of days match
fund_return = pd.DataFrame()
filt_funds = funds[(funds['DATE']>='12.04.2017') & (funds['DATE']<'31.01.2020')].drop(['DATE'],axis=1)

for fund in filt_funds.columns:
        print(fund)
        fund_return[fund] = np.log(filt_funds[fund]/
                                   filt_funds[fund].shift())

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




#%%
con_col = pd.concat([returns_train.reset_index(drop=True),
                fx_returns_train.reset_index(drop=True), 
                fund_return.reset_index(drop=True)], axis=1)

return_cov = con_col.cov()

#%%
#cov verður input
results_rand, weights_record_rand = pen_random_portfolios(15000, returns_train, fx_returns_train, 
                               fund_return,0.4,0.3,0.3, return_cov, 0.03)

#%%

max_sharpe_idx = np.argmax(results_rand[2])
sdp, rp = results_rand[0,max_sharpe_idx], results_rand[1,max_sharpe_idx]
max_sharpe_allocation = pd.DataFrame(weights_record_rand[max_sharpe_idx],index=con_col.columns,columns=['allocation'])
# max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
max_sharpe_allocation = max_sharpe_allocation.T

print ("-"*80)
print ("Maximum Sharpe Ratio Portfolio Allocation\n")
print ("Annualised Return:", round(rp,2))
print ("Annualised Volatility:", round(sdp,2))
print ("\n")
print (max_sharpe_allocation)

#%%

#Páskar
returns_date = returns_train.reset_index()['index']
funds_date = funds[(funds['DATE']>='12.04.2017') & (funds['DATE']<'31.12.2019')]['DATE'].reset_index().drop(['index'],axis=1)
fx_date = fx_returns_train.reset_index()['index']

for i in range(50,100):
     a = funds_date.iloc[i]-returns_date.iloc[i]
     print(i)
     print(a)

#80

#%%
# w1 = np.random.random(2)
# w1 /= np.sum(w1)
# w1 = w1 * 0.3


# w2 = np.random.random(2)
# w2 /= np.sum(w2)
# w2 = w2 * 0.3


# w3 = np.random.random(2)
# w3 /= np.sum(w3)
# w3 = w3 * 0.4


# weights = np.concatenate([w1,w2,w3])

#%%TEST
#%%

omx_test = get_data("LEQ.IC", "24/12/2019", "23/12/2020")
omx_returns = np.log(omx_test['adjclose']/ omx_test['adjclose'].shift())

#%%
port_returns = returns_test.dot(max_sharpe_allocation.T)


plt.plot(port_returns.cumsum())
plt.plot(omx_returns.cumsum())
plt.ylabel('RETURN')
plt.title('OPT PORTFOLIO VS OMX10 (LEQ)')
plt.legend(('Anal Bois', 'shitty market'),
           loc='upper left')


#%%
"""
Vigtuð summa til þess að neyða í ákveðið allocation

"""

weights1 = np.random.random(5)
weights1 /= np.sum(weights1)*0.3

weights2 = np.random.random(5)
weights2 /= np.sum(weights2)*0.7


# returns.mean()*252 annualize log returns
# returns.std()*np.sqrt(252) annualize stedv
# 100*(1+returns_train['eik.ic']).cumprod()
# (100*(1+returns_train).cumprod()).plot()