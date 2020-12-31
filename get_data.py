# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 14:46:52 2020

@author: Benedikt
"""

import requests
import pandas as pd
import yfinance as yf

from bs4 import BeautifulSoup
from firebase import firebase


def read_firebase(ticker):
    
    dsn = 'https://portfolio-optimization-33bbe-default-rtdb.firebaseio.com/'
    fb = firebase.FirebaseApplication(dsn, None)
    
    res = fb.get('/ticker/' + ticker, None)

    
    l = []
    for key in res:
        l.append(list(res[key].values()))
    
    out = pd.DataFrame(l, columns=[ticker, 'Date'])

    out[ticker] = out[ticker].astype(float)
    out['Date'] = pd.to_datetime(out['Date'])

    return out.drop_duplicates()


def get_omx():
    
    return(read_firebase('OMXI10'))
        
    

def get_iv():
    
    first = True
    
    urls = ['https://www.iv.is/is/sjodir/iv-sjodir/raw/11',
            'https://www.iv.is/is/sjodir/iv-sjodir/raw/10',
            'https://www.iv.is/is/sjodir/iv-sjodir/raw/9',
            # 'https://www.iv.is/is/sjodir/iv-sjodir/raw/8', no work
            'https://www.iv.is/is/sjodir/iv-sjodir/raw/1',
            'https://www.iv.is/is/sjodir/iv-sjodir/raw/20',
            'https://www.iv.is/is/sjodir/iv-sjodir/raw/21',
            'https://www.iv.is/is/sjodir/iv-sjodir/raw/16',
            'https://www.iv.is/is/sjodir/iv-sjodir/raw/19',]
    
    iv = pd.DataFrame()
    
    for url in urls:

        html_content = requests.get(url).text

        soup = BeautifulSoup(html_content, "lxml")
        
        gdp_table = soup.find("table", attrs={"class": "table"})
        title = soup.find("div", attrs={"class": "boxTitle"}).text
        
        date, price = [], []
        
        for tr in gdp_table.tbody.find_all("tr"):
        
            temp_date = tr.find_all('td')[0].text
            date.append(temp_date[6:] + '-'
                        + temp_date[3:5] + '-'
                        + temp_date[0:2])
            price.append(float(tr.find_all('td')[1].text))
        
        temp = pd.DataFrame(list(zip(date, price)), 
                       columns =['Date', title]) 
        
        if first:
            iv = temp
            first = False
        else:
            iv = pd.merge(iv, temp, on='Date', how='outer')
    
    return iv.set_index('Date').sort_index()


def get_stocks():
    
    stocks = ['BRIM.IC','ICESEA.IC','KVIKA.IC','MAREL.IC','FESTI.IC','REGINN.IC',
              'ARION.IC','VIS.IC','SJOVA.IC','ORIGO.IC','EIM.IC','HAGA.IC','EIK.IC',
              'REITIR.IC','SIMINN.IC','TM.IC','ICEAIR.IC','SKEL.IC','SYN.IC']

    stocks_df = pd.DataFrame() 
    
    for s in stocks:
        
        t = yf.Ticker(s)
        t = t.history(period='max')['Close']
        
        temp_df = t.to_frame(name=s)
        stocks_df = pd.concat([stocks_df, temp_df], axis=1)
    
    return stocks_df

def get_fx():
    tickers = ['ISK=X', 'GBPISK=X', 'EURISK=X', 'JPYISK=X', 'CHFISK=X',
               'DKKISK=X', 'NOKISK=X', 'SEKISK=X']

    fx_df = pd.DataFrame() 
    
    for s in tickers:
        
        t = yf.Ticker(s)
        t = t.history(period='max')['Close']
        
        temp_df = t.to_frame(name=s)
        fx_df = pd.concat([fx_df, temp_df], axis=1)
    
    return fx_df


def get_lb():
    pass


def get_is():
    
    first = True
    
    tickers = ['IS-LAUSAFJARSAFN', 'IS-VELTUSAFN', 'IS-RIKISSAFN',
               'IS-RIKISSKULDABREF-MEDALLONG', 'IS-RIKISSKULDABREF-LONG',
               'IS-RIKISSKULDABREF-OVERDTRYGGD', 'IS-SERTRYGGD-SKULDABREF',
               'IS-SERTRYGGD-SKULDABREF-VTR', 'IS-GRAEN-SKULDABREF-',
               'IS-HLUTABREFASJODURINN', 'IS-URVALSVISITOLUSJODURINN',
               'IS-HEIMSSAFN', 'IS-EQUUS-HLUTABREF'
               ]
    
    for ticker in tickers:
        
        temp_df = read_firebase(ticker)
        
        if first:
            is_df = temp_df
            first = False
        else:
            is_df = pd.merge(is_df, temp_df, on='Date', how='outer')
        
    is_df = is_df.set_index('Date').sort_index()
    
    return is_df


def get_ab():
    pass


# %% EXTRA
    
#   MÁ ALLS EKKI KEYRA !

'''
def read_data(): # 19:25
    
    dsn = 'https://portfolio-optimization-33bbe-default-rtdb.firebaseio.com/'
    fb = firebase.FirebaseApplication(dsn, None)
    
    urls = ['is-lausafjarsafn',
            'is-veltusafn',
            'is-rikissafn',
            'is-rikisskuldabref-medallong',
            'is-rikisskuldabref-long',
            # 'is-skuldabrefasafn',
            'is-rikisskuldabref-overdtryggd',
            'is-sertryggd-skuldabref',
            'is-sertryggd-skuldabref-vtr',
            'is-graen-skuldabref-',
            'is-hlutabrefasjodurinn',
            'is-urvalsvisitolusjodurinn',
            'is-heimssafn',
            'is-equus-hlutabref',
            ]
    
    for url in urls:
        
        print(url)

        xls = pd.read_html(url + '.xls')[0]
        xls.DateTime = xls.DateTime.apply(lambda x: x[0:10])
        
        print(type(xls))
        
        # xls = xls.rename(columns=['Date','Close'])
        
        
        for idx, row in xls.iterrows():
            data = {
                'Date' : row[0],
                'Close' : row[1]
                }
            fb.post('/ticker/' + url.upper(), data)
'''

