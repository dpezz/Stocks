import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Rest of your imports...
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import ta

# Set page config
st.set_page_config(page_title="Advanced Stock Analyzer", layout="wide")


# Custom CSS to style the app
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 18px;
        color: #333;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0366d6;
    }
    .news-item {
        background-color: #ffffff;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .news-item h4 {
        margin-top: 0;
        color: #1E1E1E;
        font-size: 16px;
        margin-bottom: 5px;
    }
    .news-item p {
        color: #333333;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .news-item a {
        color: #0366d6;
        text-decoration: none;
        font-weight: bold;
    }
    .news-item a:hover {
        text-decoration: underline;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="max")
    return history

@st.cache_data
def get_news(ticker, days=30):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    articles = newsapi.get_everything(q=ticker,
                                      from_param=start_date.strftime('%Y-%m-%d'),
                                      to=end_date.strftime('%Y-%m-%d'),
                                      language='en',
                                      sort_by='publishedAt')
    return articles['articles']

def calculate_sentiment(news):
    sentiments = []
    for article in news:
        title = article.get('title', '')
        description = article.get('description', '')
        if title or description:
            text = f"{title} {description}".strip()
            sentiment = TextBlob(text).sentiment.polarity
            sentiments.append(sentiment)
    return np.mean(sentiments) if sentiments else 0

def prepare_data(data, news):
    data['Date'] = data.index
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])

    data['Returns'] = data['Close'].pct_change()

    sentiment_scores = []
    for date in data.index:
        relevant_news = [article for article in news if article.get('publishedAt') and date.strftime('%Y-%m-%d') in article['publishedAt']]
        sentiment_scores.append(calculate_sentiment(relevant_news))
    data['Sentiment'] = sentiment_scores

    data = data.ffill()

    return data.dropna()

def train_model(data):
    features = ['Days', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'ATR', 'Returns', 'Volume', 'Sentiment']
    X = data[features]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, features

def predict_future(model, data, features, days=30):
    last_day = data['Days'].max()
    future_data = pd.DataFrame({'Days': range(last_day + 1, last_day + days + 1)})
    
    for feature in features:
        if feature != 'Days':
            future_data[feature] = data[feature].tail(days).values
    
    predictions = model.predict(future_data[features])
    return future_data, predictions

def plot_news_impact(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Stock Price', 'Sentiment Score'))
    
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Stock Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Sentiment'], name='Sentiment Score'), row=2, col=1)
    
    fig.update_layout(height=600, title_text="News Sentiment Impact on Stock Price")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1)
    
    return fig

def plot_volume_analysis(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Stock Price', 'Trading Volume'))
    
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Stock Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
    
    # Calculate and plot moving average of volume
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=data.index, y=data['Volume_MA'], name='20-day Volume MA', line=dict(color='red')), row=2, col=1)
    
    fig.update_layout(height=600, title_text="Stock Price and Trading Volume Analysis")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def main():
    st.title("📈 Advanced Stock Analyzer")

    with st.sidebar:
        st.header("Welcome!")
        ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()

    if ticker:
        stock_data = get_stock_data(ticker)
        news = get_news(ticker)
        
        prepared_data = prepare_data(stock_data, news)
        
        model, X_test, y_test, features = train_model(prepared_data)

        future_data, predictions = predict_future(model, prepared_data, features)

        st.sidebar.markdown("---")
        st.sidebar.header("Key Metrics")

        current_price = prepared_data['Close'].iloc[-1]
        predicted_price = predictions[-1]
        price_diff = predicted_price - current_price
        price_change_percent = (price_diff / current_price) * 100

        st.sidebar.markdown(f"""
        <div class="metric-card">
            <h3>Current Price</h3>
            <div class="metric-value">${current_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        color = "green" if price_change_percent > 0 else "red"
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Price (30 days)</h3>
            <div class="metric-value" style="color: {color};">${predicted_price:.2f}</div>
            <div style="color: {color};">({price_change_percent:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

        mse = mean_squared_error(y_test, model.predict(X_test))
        rmse = np.sqrt(mse)
        avg_price = prepared_data['Close'].mean()
        rmse_percentage = (rmse / avg_price) * 100

        st.sidebar.markdown(f"""
        <div class="metric-card">
            <h3>Model Accuracy</h3>
            <div class="metric-value">${rmse:.2f}</div>
            <div>({rmse_percentage:.2f}% of avg. price)</div>
            <div class="tooltip">ℹ️
                <span class="tooltiptext">
                    This is the average prediction error (RMSE).
                    Lower values indicate better accuracy.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "📰 News Impact", "📉 Volume Analysis"])

        with tab1:
            st.header(f"{ticker} Stock Analysis")
            st.markdown(f"Analyzing data from {stock_data.index[0].date()} to {stock_data.index[-1].date()}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prepared_data.index, y=prepared_data['Close'], name='Historical Data'))
            fig.add_trace(go.Scatter(x=pd.date_range(start=prepared_data.index[-1], periods=31)[1:], 
                                     y=predictions, name='Predictions', line=dict(dash='dash')))
            fig.update_layout(title=f"{ticker} Stock Price and Predictions",
                              xaxis_title="Date",
                              yaxis_title="Price",
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Technical Indicators"):
                col1, col2 = st.columns(2)
                
                rsi_value = prepared_data['RSI'].iloc[-1]
                rsi_color = "green" if 30 <= rsi_value <= 70 else "red"
                col1.markdown(f"""
                <div>
                    RSI: <span style="color:{rsi_color}; font-weight:bold;">{rsi_value:.2f}</span>
                    <span class="tooltip">ℹ️
                        <span class="tooltiptext">
                            RSI (Relative Strength Index) measures momentum. 
                            Values over 70 suggest overbought conditions, 
                            while values under 30 suggest oversold conditions.
                        </span>
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                macd_value = prepared_data['MACD'].iloc[-1]
                macd_color = "green" if macd_value > 0 else "red"
                col2.markdown(f"""
                <div>
                    MACD: <span style="color:{macd_color}; font-weight:bold;">{macd_value:.2f}</span>
                    <span class="tooltip">ℹ️
                        <span class="tooltiptext">
                            MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator.
                            Positive values suggest bullish momentum, while negative values suggest bearish momentum.
                        </span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.header("News Sentiment Impact")
            news_impact_fig = plot_news_impact(prepared_data)
            st.plotly_chart(news_impact_fig, use_container_width=True)

            st.subheader("Recent News")
            for article in news[:5]:
                sentiment = TextBlob(article.get('title', '') + ' ' + article.get('description', '')).sentiment.polarity
                sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
                st.markdown(f"""
                <div class="news-item">
                    <h4>{article.get('title', 'No title')}</h4>
                    <p>{article.get('description', 'No description')}</p>
                    <p>Sentiment: <span style="color:{sentiment_color};">{sentiment:.2f}</span></p>
                    <a href="{article.get('url', '#')}" target="_blank">Read more</a>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.header("Trading Volume Analysis")
            volume_analysis_fig = plot_volume_analysis(prepared_data)
            st.plotly_chart(volume_analysis_fig, use_container_width=True)

            with st.expander("Volume Statistics"):
                avg_volume = prepared_data['Volume'].mean()
                max_volume = prepared_data['Volume'].max()
                st.metric("Average Daily Volume", f"{avg_volume:,.0f}")
                st.metric("Max Volume", f"{max_volume:,.0f}")

if __name__ == "__main__":
    main()