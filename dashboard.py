import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

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
    
    return stocks_df[(stocks_df.index > pd.to_datetime(end_date)) & (stocks_df.index < pd.to_datetime(start_date))]

st.sidebar.title("Data Selector")
# st.sidebar.markdown("Velja tikker:")

stocks = ['BRIM.IC','ICESEA.IC','KVIKA.IC','MAREL.IC','FESTI.IC','REGINN.IC',
          'ARION.IC','VIS.IC','SJOVA.IC','ORIGO.IC','EIM.IC','HAGA.IC','EIK.IC',
          'REITIR.IC','SIMINN.IC','TM.IC','ICEAIR.IC','SKEL.IC','SYN.IC']



# st.write(data)
# st.line_chart(data)

import datetime

today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=-14)
end_date = st.sidebar.date_input('Start date', tomorrow)
start_date = st.sidebar.date_input('End date', today)

    
    
options = st.sidebar.multiselect('Ticker(s):', stocks, stocks[0])

data_load_state = st.text('Loading data...')
# data = load_data(10000)
data = get_stocks(start_date, end_date, options)
# st.write(data)
data_load_state.text("Done! (using st.cache)")
    
# data = get_stocks(start_date, end_date, options)
st.line_chart(data)
st.line_chart(data.pct_change())
