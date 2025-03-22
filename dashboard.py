import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

from matplotlib import pyplot as plt
from get_data import get_omx, get_iv, get_is
from opt_functions import pen_random_portfolios2
from datetime import timedelta 

st.title('Icelandic Stock Exchange')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


@st.cache()
def get_stocks(start_date, end_date, stocks):

    stocks_df = pd.DataFrame() 
    
    for s in stocks:
        
        t = yf.Ticker(s)
        t = t.history(period='max')['Close']
        
        temp_df = t.to_frame(name=s)
        stocks_df = pd.concat([stocks_df, temp_df], axis=1)

    stocks_df.index = pd.to_datetime(stocks_df.index)
    print(type(stocks_df.index))
    print(type(pd.to_datetime(start_date)))

    return stocks_df[(stocks_df.index.tz_convert(None) > pd.to_datetime(end_date)) & (stocks_df.index.tz_convert(None) < pd.to_datetime(start_date))]

st.sidebar.title("Data Selector")
# st.sidebar.markdown("Velja tikker:")

stocks = ['BRIM.IC','ICESEA.IC','KVIKA.IC',
          #'MAREL.IC',
          'FESTI.IC',#'REGINN.IC',
          'ARION.IC','VIS.IC','SJOVA.IC','ORIGO.IC','EIM.IC','HAGA.IC','EIK.IC',
          'REITIR.IC','SIMINN.IC','TM.IC','ICEAIR.IC','SKEL.IC','SYN.IC']


# st.write(data)
# st.line_chart(data)

import datetime

today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=-90)
end_date = st.sidebar.date_input('Start date', tomorrow)
start_date = st.sidebar.date_input('End date', today)

    
    
options = st.sidebar.multiselect('Ticker(s):', stocks, stocks)

data_load_state = st.text('Loading data...')
# data = load_data(10000)
data = get_stocks(start_date, end_date, options)
# st.write(data)
data_load_state.text("Done! (using st.cache)")
    
# data = get_stocks(start_date, end_date, options)


fig = px.line(data, x=data.index, y=data.columns,
              title='custom tick labels')


# st.line_chart(data)
st.plotly_chart(fig)
# st.line_chart(data.pct_change())

train_returns = np.log(data/data.shift())
cov_matrix = train_returns.cov()
num_sims = 25000
# w_iv = 0.2
w_stocks = 1.0
# w_is = 0.2

rf = 0.03

# iv_returns_train = train_returns[['ÍV Erlent hlutabréfasafn - Runugildi',
                          # 'ÍV Alþjóðlegur hlutabréfasjóður - Runugildi']]

stocks_returns_train = train_returns[stocks]

# is_return_trains = train_returns[['IS-RIKISSKULDABREF-LONG',
                                     # 'IS-SERTRYGGD-SKULDABREF']]

# results_rand, weights_rand = pen_random_portfolios(num_sims, iv_returns_train, 
                                                   # stocks_returns_train, is_return_trains,
                                                   # w_iv, w_stocks, w_is, cov_matrix, rf)
results_rand, weights_rand = pen_random_portfolios2(num_sims, stocks_returns_train, w_stocks, cov_matrix, rf)



fig, ax = plt.subplots()
cm = plt.cm.get_cmap('RdYlBu')
# sc = plt.scatter(results_rand[0], results_rand[1], c=results_rand[2], cmap=cm)
sc = px.scatter(x=results_rand[0], y=results_rand[1], color=results_rand[2])
# plt.colorbar(sc)
# plt.xlabel('RISK')
# plt.ylabel('EXPECTED RETURN')
# plt.title('MONTE CARLO PORTFOLIO SIMULATIONS')


max_sharpe_idx = np.argmax(results_rand[2])
sharpe_value = results_rand[2,max_sharpe_idx]
sdp, rp = results_rand[0,max_sharpe_idx], results_rand[1,max_sharpe_idx]

# plt.scatter(sdp,rp, s=400, marker='x', c='orange')
px.scatter(x=np.array([sdp]), y=np.array([rp]))

max_sharpe_allocation = pd.DataFrame(weights_rand[max_sharpe_idx],index=cov_matrix.columns,columns=['allocation'])
max_sharpe_allocation = max_sharpe_allocation.T
plt.show()

st.plotly_chart(sc)