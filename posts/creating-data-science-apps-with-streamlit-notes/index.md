---
categories:
- python
- streamlit
- numpy
- pandas
- notes
date: 2022-1-2
description: My notes from Chanin Nantasenamat's video on creating data science web
  apps with Streamlit.
hide: false
search_exclude: false
title: Notes on Creating Data Science Apps With Streamlit

aliases:
- /Notes-on-Creating-Data-Science-Apps-With-Streamlit/
---

* [Overview](#overview)
* [Streamlit](#)
* [Simple Stock Price](#simple-stock-price)
* [Simple Bioinformatics DNA Count](#simple-bioinformatics-dna-count)
* [EDA Basketball](#eda-basketball)
* [EDA Cryptocurrency](#eda-cryptocurrency)
* [Classification Iris Data](#classification-iris-data)
* [Regression Boston Housing Data](#regression-boston-housing-data)
* [Deploy App to Heroku](#deploy-app-to-heroku)
* [Deploy App to Streamlit Sharing](#deploy-app-to-streamlit-sharing)



## Overview

Here are some notes I took while watching Chanin Nantasenamat's [video](https://www.youtube.com/watch?v=JwSS70SZdyM) on creating data science web apps with Streamlit.


## Streamlit
* [Streamlit - The fastest way to build and share data apps](https://streamlit.io/)

- Turns data scripts into shareable web apps
- `pip install streamlit`
- Test Installation: `streamlit hello`
- Run apps: `streamlit run main.py`
- Format text using [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)







## Simple Stock Price

Get market data from Yahoo! Finance API

- [yfinance python package](https://pypi.org/project/yfinance/)
- [How to Get Stock Data Using Python](https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75)

**Dependencies**

- [Pandas](https://pandas.pydata.org/)
    - `pip install pandas`
- [Streamlit](https://streamlit.io/)
    - `pip install streamlit`
- [yfinance](https://github.com/ranaroussi/yfinance)
  - `pip install yfinance`



**replit:** [Simple_Stock_Price](https://replit.com/@innominate817/SimpleStockPrice#main.py)

```python
import yfinance as yf
import streamlit as st
import pandas as pd

# Write text in Markdown format
st.write("""
# Simple Stock Price App

Shown are the stock closing price and volume of iPath Global Carbon ETN!

""")

# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
# define the ticker symbol
tickerSymbol = 'GRN'
# get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# get the historical pricess for this ticker
# Open High Low Close Volume Dividends Stock Splits
tickerDf = tickerData.history(period='1d', start='2019-12-27', end='2021-12-27')

# Create streamlit line charts
st.write("""
## Closing Price
""")
st.line_chart(tickerDf.Close)
st.write("""
## Trading Volume
""")
st.line_chart(tickerDf.Volume)
```




## Simple Bioinformatics DNA Count

Count the number of nucleotides `'A', 'T', 'G', 'C'` in entered in a text box

**Dependencies**

- [Pandas](https://pandas.pydata.org/)
    - `pip install pandas`
- [Streamlit](https://streamlit.io/)
    - `pip install streamlit`
- [Altair](https://altair-viz.github.io/)
    - `pip install altair`
- [Pillow](https://pillow.readthedocs.io/en/stable/)
    - `pip install pillow`

**replit:** [Simple_Bioinformatics_DNA_Count](https://replit.com/@innominate817/SimpleBioinformaticsDNACount#main.py)

```python
# Import dependencies
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

# Page Title
# Add hero image
image = Image.open('dna-ge3ed05159_1920.jpg')
st.image(image, use_column_width=True)

st.write("""
# DNA Nucleotide Count Web App
This app counts the nucleotide composition of query DNA!

***
""")

# Input Text Box
#st.sidebar.header('Enter DNA sequence')
st.header('Enter DNA sequence')

sequence_input = ">DNA Query\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"

sequence = st.text_area("Sequence input", sequence_input, height=250)
# Split input text by line
sequence = sequence.splitlines()
# Skip the sequence name (first line)
sequence = sequence[1:]
# Concatenate list to string
sequence = ''.join(sequence)

st.write("""
***
""")

# Print the input DNA sequence
st.header('INPUT (DNA Query)')
sequence

# DNA nucleotide count
st.header('OUTPUT (DNA Nucleotide Count)')

# 1. Print dictionary
st.subheader('1. Prince dictionary')
def DNA_nucleotide_count(seq):
  d = dict([
    ('A', seq.count('A')),
    ('T', seq.count('T')),
    ('G', seq.count('G')),
    ('C', seq.count('C'))
  ])
  return d

X = DNA_nucleotide_count(sequence)

X

# 2. Print text
st.subheader('2. Print text')
st.write('There are ' + str(X['A']) + ' adenine (A)')
st.write('There are ' + str(X['T']) + ' thymine (T)')
st.write('There are ' + str(X['G']) + ' guanine (G)')
st.write('There are ' + str(X['C']) + ' cytosine (C)')

# 3. Display DataFrame
st.subheader('3. Display DataFrame')
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0: 'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'nucleotide'})
st.write(df)

# 4. Display Bar Chart using Altair
st.subheader('4. Display Bar chart')
p = alt.Chart(df).mark_bar().encode(
  x='nucleotide',
  y='count'
)

p = p.properties(
  # Controls width of bar
  width=alt.Step(80)
)
st.write(p)
```






## EDA Basketball

Scrape NBA player stats from a website and perform exploratory data analysis.

**Dependencies**

- [Pandas](https://pandas.pydata.org/)
    - `pip install pandas`
- [Streamlit](https://streamlit.io/)
    - `pip install streamlit`
- [Matplotlib](https://matplotlib.org/stable/index.html#)
    - `pip install matplotlib`
- [Seaborn](https://seaborn.pydata.org/index.html)
    - `pip install seaborn`
- [Numpy](https://numpy.org/)
    - `pip install numpy`
- [lxml](https://lxml.de/)
    - `pip install lxml`

**Data Source**

[Basketball Statistics and History](https://www.basketball-reference.com/)

**replit:** [EDA_Basketball](https://replit.com/@innominate817/EDABasketball#main.py)

```python
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA Player Stats Explorer')

st.markdown("""
This app performs simple websraping of NBA player stats data!
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/)
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2020))))

# Web scraping of NBA player stats
@st.cache
def load_data(year):
  url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html'
  html = pd.read_html(url, header=0)
  df = html[0]
  # Delete repeating headers
  raw = df.drop(df[df.Age == 'Age'].index)
  # Fill missing data with 0
  raw = raw.fillna(0)
  # Convert int columns to float
  raw['FG%'] = raw['FG%'].astype(float)
  raw['3P%'] = raw['3P%'].astype(float)
  raw['2P%'] = raw['2P%'].astype(float)
  raw['eFG%'] = raw['eFG%'].astype(float)
  raw['FT%'] = raw['FT%'].astype(float)
  # Remove redundant index column
  playerstats = raw.drop(['Rk'], axis=1)
  return playerstats

playerstats = load_data(selected_year)

# sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download NBA player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
  csv = df.to_csv(index=False)
  # strings <-> bytes conversion
  b64 = base64.b64encode(csv.encode()).decode()
  href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
  return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
  st.header('Intercorrelation Matrix Heatmap')
  df_selected_team.to_csv('output.csv', index=False)
  df = pd.read_csv('output.csv')

  corr = df.corr()
  mask = np.zeros_like(corr)
  mask[np.triu_indices_from(mask)] = True
  fig = None
  with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(7,5))
    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
  st.pyplot(fig)
```




## EDA Cryptocurrency

Use the BeautifulSoup library to scrape data from [CoinMarketCap](https://coinmarketcap.com) and perform exploratory data analysis.

**Dependencies**

- [Pandas](https://pandas.pydata.org/)
    - `pip install pandas`
- [Streamlit](https://streamlit.io/)
    - `pip install streamlit`
- [Matplotlib](https://matplotlib.org/stable/index.html#)
    - `pip install matplotlib`
- [lxml](https://lxml.de/)
    - `pip install lxml`
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
    - `pip install beautifulsoup4`

**replit:** [EDA_Cryptocurrency](https://replit.com/@innominate817/EDACryptocurrency#main.py)

```python
# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import json
import time

# ---------------------------------#
# New feature (make sure to upgrade your streamlit library)
# pip install --upgrade streamlit

# ---------------------------------#
# Page layout
# Page expands to full width
st.set_page_config(layout="wide")
# ---------------------------------#
# Title

image = Image.open("pexels-worldspectrum-844124.jpg")

st.image(image, width=500)

st.title("Crypto Price App")
st.markdown(
    """
This app retrieves cryptocurrency prices for the top 100 cryptocurrency from the **CoinMarketCap**!

"""
)
# ---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown(
    """
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
* **Data source:** [CoinMarketCap](http://coinmarketcap.com).
* **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
"""
)

# ---------------------------------#
# Page layout (continued)
# Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((2, 1))

# ---------------------------------#
# Sidebar + Main panel
col1.header("Input Options")

# Sidebar - Currency price unit
currency_price_unit = col1.selectbox("Select currency for price", ("USD", "BTC", "ETH"))

# Web scraping of CoinMarketCap data
@st.cache
def load_data():
    cmc = requests.get("https://coinmarketcap.com")
    soup = BeautifulSoup(cmc.content, "html.parser")

    data = soup.find("script", id="__NEXT_DATA__", type="application/json")
    coins = {}
    coin_data = json.loads(data.contents[0])
    listings = coin_data["props"]["initialState"]["cryptocurrency"]["listingLatest"][
        "data"
    ]

    attributes = listings[0]["keysArr"]
    index_of_id = attributes.index("id")
    index_of_slug = attributes.index("slug")

    for i in listings[1:]:
        coins[str(i[index_of_id])] = i[index_of_slug]

    coin_name = []
    coin_symbol = []
    market_cap = []
    percent_change_1h = []
    percent_change_24h = []
    percent_change_7d = []
    price = []
    volume_24h = []

    index_of_slug = attributes.index("slug")
    index_of_symbol = attributes.index("symbol")

    index_of_quote_currency_price = attributes.index(
        f"quote.{currency_price_unit}.price"
    )
    index_of_quote_currency_percent_change_1h = attributes.index(
        f"quote.{currency_price_unit}.percentChange1h"
    )
    index_of_quote_currency_percent_change_24h = attributes.index(
        f"quote.{currency_price_unit}.percentChange24h"
    )
    index_of_quote_currency_percent_change_7d = attributes.index(
        f"quote.{currency_price_unit}.percentChange7d"
    )
    index_of_quote_currency_market_cap = attributes.index(
        f"quote.{currency_price_unit}.marketCap"
    )
    index_of_quote_currency_volume_24h = attributes.index(
        f"quote.{currency_price_unit}.volume24h"
    )

    for i in listings[1:]:
        coin_name.append(i[index_of_slug])
        coin_symbol.append(i[index_of_symbol])

        price.append(i[index_of_quote_currency_price])
        percent_change_1h.append(i[index_of_quote_currency_percent_change_1h])
        percent_change_24h.append(i[index_of_quote_currency_percent_change_24h])
        percent_change_7d.append(i[index_of_quote_currency_percent_change_7d])
        market_cap.append(i[index_of_quote_currency_market_cap])
        volume_24h.append(i[index_of_quote_currency_volume_24h])

    df = pd.DataFrame(
        columns=[
            "coin_name",
            "coin_symbol",
            "market_cap",
            "percent_change_1h",
            "percent_change_24h",
            "percent_change_7d",
            "price",
            "volume_24h",
        ]
    )
    df["coin_name"] = coin_name
    df["coin_symbol"] = coin_symbol
    df["price"] = price
    df["percent_change_1h"] = percent_change_1h
    df["percent_change_24h"] = percent_change_24h
    df["percent_change_7d"] = percent_change_7d
    df["market_cap"] = market_cap
    df["volume_24h"] = volume_24h
    return df

df = load_data()

# Sidebar - Cryptocurrency selections
sorted_coin = sorted(df["coin_symbol"])
selected_coin = col1.multiselect("Cryptocurrency", sorted_coin, sorted_coin)

df_selected_coin = df[(df["coin_symbol"].isin(selected_coin))]  # Filtering data

# Sidebar - Number of coins to display
num_coin = col1.slider("Display Top N Coins", 1, 100, 100)
df_coins = df_selected_coin[:num_coin]

# Sidebar - Percent change timeframe
percent_timeframe = col1.selectbox("Percent change time frame", ["7d", "24h", "1h"])
percent_dict = {
    "7d": "percent_change_7d",
    "24h": "percent_change_24h",
    "1h": "percent_change_1h",
}
selected_percent_timeframe = percent_dict[percent_timeframe]

# Sidebar - Sorting values
sort_values = col1.selectbox("Sort values?", ["Yes", "No"])

col2.subheader("Price Data of Selected Cryptocurrency")
col2.write(
    "Data Dimension: "
    + str(df_selected_coin.shape[0])
    + " rows and "
    + str(df_selected_coin.shape[1])
    + " columns."
)

col2.dataframe(df_coins)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href

col2.markdown(filedownload(df_selected_coin), unsafe_allow_html=True)

# ---------------------------------#
# Preparing data for Bar plot of % Price change
col2.subheader("Table of % Price Change")
df_change = pd.concat(
    [
        df_coins.coin_symbol,
        df_coins.percent_change_1h,
        df_coins.percent_change_24h,
        df_coins.percent_change_7d,
    ],
    axis=1,
)
df_change = df_change.set_index("coin_symbol")
df_change["positive_percent_change_1h"] = df_change["percent_change_1h"] > 0
df_change["positive_percent_change_24h"] = df_change["percent_change_24h"] > 0
df_change["positive_percent_change_7d"] = df_change["percent_change_7d"] > 0
col2.dataframe(df_change)

# Conditional creation of Bar plot (time frame)
col3.subheader("Bar plot of % Price Change")

if percent_timeframe == "7d":
    if sort_values == "Yes":
        df_change = df_change.sort_values(by=["percent_change_7d"])
    col3.write("*7 days period*")
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change["percent_change_7d"].plot(
        kind="barh",
        color=df_change.positive_percent_change_7d.map({True: "g", False: "r"}),
    )
    col3.pyplot(plt)
elif percent_timeframe == "24h":
    if sort_values == "Yes":
        df_change = df_change.sort_values(by=["percent_change_24h"])
    col3.write("*24 hour period*")
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change["percent_change_24h"].plot(
        kind="barh",
        color=df_change.positive_percent_change_24h.map({True: "g", False: "r"}),
    )
    col3.pyplot(plt)
else:
    if sort_values == "Yes":
        df_change = df_change.sort_values(by=["percent_change_1h"])
    col3.write("*1 hour period*")
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change["percent_change_1h"].plot(
        kind="barh",
        color=df_change.positive_percent_change_1h.map({True: "g", False: "r"}),
    )
    col3.pyplot(plt)
```





## Classification Iris Data

Use scikit-learn to perform classification with a Random Forest Classifier.

**Dependencies**

- [Pandas](https://pandas.pydata.org/)
    - `pip install pandas`
- [Streamlit](https://streamlit.io/)
    - `pip install streamlit`
- [Scikit learn](https://scikit-learn.org/stable/)
    - `pip install scikit-learn`

**replit:** [Classification_Iris_Data](https://replit.com/@innominate817/ClassificationIrisData#main.py)

```python
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** typ:
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
```





## Regression Boston Housing Data

Use regression to predict housing prices.

**Dependencies**

- [Pandas](https://pandas.pydata.org/)
    - `pip install pandas`
- [Streamlit](https://streamlit.io/)
    - `pip install streamlit`
- [Scikit learn](https://scikit-learn.org/stable/)
    - `pip install scikit-learn`
- [shap](https://github.com/slundberg/shap)
    - `pip install shap`
- [Matplotlib](https://matplotlib.org/stable/index.html#)
    - `pip install matplotlib`

**replit:** [Regression_Boston_Housing_Data](https://replit.com/@innominate817/RegressionBostonHousingData#main.py)

```python
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

fig, ax = plt.subplots()

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(fig, bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(fig, bbox_inches='tight')
```



## Deploy App to Heroku

[Heroku](https://www.heroku.com/)

`runtime.txt`

- contains the required python version

```python
python-3.7.9
```

`requirements.txt`

- contains the required packages and version numbers

```python
streamlit==0.61.0
pandas==0.25.3
numpy==1.18.1
scikit-learn==0.22.1
```

`setup.sh`

- contains the setup steps for the server on the Heroku dyno

```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```

`Procfile`

- runs the `[setup.sh](http://setup.sh)` file and starts the streamlit app

```bash
web: sh setup.sh && streamlit run app.py
```





## Deploy App to Streamlit Sharing

[Your Streamlit Apps](https://share.streamlit.io/)

[Streamlit Cloud Example Apps](https://share.streamlit.io/streamlit/cloud-example-apps/main)

`requirements.txt`

- contains the required packages and version numbers

```python
streamlit==0.61.0
pandas==0.25.3
numpy==1.18.1
scikit-learn==0.22.1
```






**References:**

* [Build 12 Data Science Apps with Python and Streamlit - Full Course](https://www.youtube.com/watch?v=JwSS70SZdyM)



<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->