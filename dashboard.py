import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import datetime
import matplotlib.pyplot as plt


from opt_functions import pen_random_portfolios2


@st.cache_data()
def get_stocks(start_date, end_date, stocks):

    stocks_df = pd.DataFrame() 
    
    for s in stocks:
        
        t = yf.Ticker(s)
        t = t.history(period='max')['Close']
        
        temp_df = t.to_frame(name=s)
        stocks_df = pd.concat([stocks_df, temp_df], axis=1)

    stocks_df.index = pd.to_datetime(stocks_df.index)
    return stocks_df[(stocks_df.index.tz_convert(None) > pd.to_datetime(end_date))
                     & (stocks_df.index.tz_convert(None) < pd.to_datetime(start_date))]


st.sidebar.title("Data Selector")

stocks = ['ALVO.IC','AMRQ.IC','ARION.IC','BRIM.IC','EIK.IC','EIM.IC','FESTI.IC','HAGA.IC','HAMP.IC','HEIMAR.IC',
          'ICEAIR.IC','ICESEA.IC','ISB.IC','ISF.IC','JBTM.IC','KALD.IC','KVIKA.IC','NOVA.IC','CS.IC','OLGERD.IC',
          'PLAY.IC','REITIR.IC','SIMINN.IC','SJOVA.IC','SKAGI.IC','SKEL.IC','SVN.IC','SYN.IC']

today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=-90)
end_date = st.sidebar.date_input('Start date', tomorrow)
start_date = st.sidebar.date_input('End date', today)
options = st.sidebar.multiselect('Ticker(s):', stocks, stocks)


data_load_state = st.text('Loading data...')
data = get_stocks(start_date, end_date, options)
data_load_state.text("Done! (using st.cache)")


def plot_historical_stock_prices():
    fig = px.line(data, x=data.index, y=data.columns,
                  title='Stock prices')

    st.plotly_chart(fig)


def plot_monte(options):
    # Calculate returns and covariance
    train_returns = np.log(data / data.shift())
    cov_matrix = train_returns.cov()
    num_sims = 100
    w_stocks = 1.0
    rf = 0.03
    stocks_returns_train = train_returns[options]

    # Generate Monte Carlo simulation results
    results_rand, weights_rand = pen_random_portfolios2(num_sims, stocks_returns_train, w_stocks, cov_matrix, rf)

    # Create the matplotlib figure
    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax.scatter(results_rand[0], results_rand[1], c=results_rand[2], cmap=cm)
    plt.colorbar(sc, ax=ax)
    ax.set_xlabel('RISK')
    ax.set_ylabel('EXPECTED RETURN')
    ax.set_title('MONTE CARLO PORTFOLIO SIMULATIONS')

    # Identify the portfolio with the maximum Sharpe ratio
    max_sharpe_idx = np.argmax(results_rand[2])
    sharpe_value = results_rand[2, max_sharpe_idx]
    sdp, rp = results_rand[0, max_sharpe_idx], results_rand[1, max_sharpe_idx]
    ax.scatter(sdp, rp, s=400, marker='x', c='orange')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Prepare the allocation table for the maximum Sharpe ratio portfolio
    max_sharpe_allocation = pd.DataFrame(
        weights_rand[max_sharpe_idx],
        index=cov_matrix.columns,
        columns=['allocation']
    ).T

    # Display simulation results
    st.write("-" * 80)
    st.write("### Maximum Sharpe Ratio Portfolio Allocation")
    st.write("**Annualised Return:**", round(rp, 2))
    st.write("**Annualised Volatility:**", round(sdp, 2))
    st.dataframe(max_sharpe_allocation)


st.title('Icelandic Stock Exchange - Portfolio Optimizer')

st.write("")
st.markdown("""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas eu quam sed sapien rhoncus semper. Vestibulum tempor dolor nibh, a eleifend purus tincidunt eu. Integer ultricies, lectus non varius rutrum, ante libero venenatis justo, sed porta augue justo id mauris. Praesent in neque vitae nisi fringilla vestibulum. Suspendisse pretium ornare finibus. Donec sit amet convallis enim. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas sagittis nulla in sem cursus imperdiet. In ornare erat id eros hendrerit aliquam. Integer suscipit lacus leo. Etiam malesuada, diam commodo porttitor tempus, elit justo pulvinar quam, eu volutpat lacus libero eu elit.
""")

plot_historical_stock_prices()

plot_monte(options)