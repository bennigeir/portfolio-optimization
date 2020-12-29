# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:47:51 2020

@author: Benedikt
"""

import yfinance as yf
import requests

from firebase import firebase
from datetime import date, timedelta
from bs4 import BeautifulSoup


def update_omx():
    
    ticker = yf.Ticker('^OMXI10')
    prev_price = ticker.info['regularMarketPreviousClose']
    
    yesterday = date.today() - timedelta(days=1)
    d1 = yesterday.strftime('%Y-%m-%d')
    
    dsn = 'https://portfolio-optimization-33bbe-default-rtdb.firebaseio.com/'
    fb = firebase.FirebaseApplication(dsn, None)
    
    data = {
        'Date' : d1,
        'Close' : prev_price
        }
        
    fb.post('/ticker/OMXI10', data)   


def update_ib():
    
    base1 = 'https://www.islandssjodir.is/sjodir/skuldabrefasjodir/'
    base2 = 'https://www.islandssjodir.is/sjodir/hlutabrefasjodir/'
    
    urls1 = ['is-lausafjarsafn',
            'is-veltusafn',
            'is-rikissafn',
            'is-rikisskuldabref-medallong',
            'is-rikisskuldabref-long',
            'is-skuldabrefasafn',
            'is-rikisskuldabref-overdtryggd',
            'is-sertryggd-skuldabref',
            'is-sertryggd-skuldabref-vtr',
            'is-graen-skuldabref-',
            ]
    
    urls2 = [
            'is-hlutabrefasjodurinn',
            'is-urvalsvisitolusjodurinn',
            'is-heimssafn',
            'is-equus-hlutabref',
            ]

    for url in urls1:

        html_content = requests.get(base1 + url).text


        soup = BeautifulSoup(html_content, "lxml")
        
        temp_price = soup.find("div", attrs={"class": "col-xs-6 fundbox__rate"}).text
        temp_date = soup.find("div", attrs={"class": "col-xs-12 fundbox__rate-date text-right"}).text
        
        temp_price = temp_price.replace('.', '').replace(',', '.')
        temp_date = temp_date.strip().replace('Gengi ', '')
        
        date = temp_date[6:] + '-' + temp_date[3:5] + '-' + temp_date[0:2]
        price = temp_price

        dsn = 'https://portfolio-optimization-33bbe-default-rtdb.firebaseio.com/'
        fb = firebase.FirebaseApplication(dsn, None)
        
        data = {
            'Date' : date,
            'Close' : price
            }
            
        fb.post('/ticker/' + url.upper(), data)
        
    
    for url in urls2:

        html_content = requests.get(base2 + url).text


        soup = BeautifulSoup(html_content, "lxml")
        
        temp_price = soup.find("div", attrs={"class": "col-xs-6 fundbox__rate"}).text
        temp_date = soup.find("div", attrs={"class": "col-xs-12 fundbox__rate-date text-right"}).text
        
        temp_price = temp_price.replace('.', '').replace(',', '.')
        temp_date = temp_date.strip().replace('Gengi ', '')
        
        date = temp_date[6:] + '-' + temp_date[3:5] + '-' + temp_date[0:2]
        price = temp_price

        dsn = 'https://portfolio-optimization-33bbe-default-rtdb.firebaseio.com/'
        fb = firebase.FirebaseApplication(dsn, None)
        
        data = {
            'Date' : date,
            'Close' : price
            }
            
        fb.post('/ticker/' + url.upper(), data)


weekno = date.today().weekday()

# if weekno in [1,2,3,4,5]:
    # update_omx()
    
# if weekno in [0,1,2,3,4]:    
    # update_ib()