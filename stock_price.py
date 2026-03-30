import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import warnings
import sys
import os
import logging

# ---------------- Basic setup and logging ----------------
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Stock Price Prediction Using Time Series Analysis", layout="wide")

# Theme colors for plots/cards
theme = {
    "background": "#0B0C10",
    "text": "#E5E7EB",
    "up": "#22C55E",
    "down": "#EF4444",
    "accent": "#FFD700",
    "card": "#1F2937"
}

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
# ---------------- STDERR SUPPRESSION (context manager) ----------------
# Use this when calling yfinance to suppress noisy stderr logs.
class SuppressYFinanceErrors:
    def __enter__(self):
        self.original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.devnull.close()
        sys.stderr = self.original_stderr
        return False

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Stock Settings")

# text_input with help (tooltip) - instruct user on format and examples
tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated):",
    "AAPL,MSFT,GOOGL,TCS,RELIANCE,GOLD1,SILVERETF",
    help="Enter tickers comma-separated. Examples: AAPL, MSFT, GOOG, TCS, RELIANCE. For some commodities use GOLD1 or SILVERETF."
).upper().replace(" ", "").split(",")

# slider with help text
refresh_rate = st.sidebar.slider(
    "⏱️ Refresh Rate (seconds)", 1, 10, 2,
    help="How often (in seconds) the live prices and UI will refresh. Lower = more frequent updates (may increase API calls)."
)
forecast_days = st.sidebar.slider(
    "🔮 Forecast Days", 1, 30, 10,
    help="Number of days into the future to forecast using the models (ARIMA and LSTM)."
)

st.sidebar.markdown("### 📅 Select Date Range")
date_range = st.sidebar.date_input(
    "Choose Date Range",
    [datetime.now() - timedelta(days=90), datetime.now()],
    help="Select the historical date range to display in the historical chart and historical table."
)
from_date, to_date = date_range if len(date_range) == 2 else (datetime.now() - timedelta(days=90), datetime.now())
if from_date > to_date:
    st.sidebar.error("❌ 'From Date' cannot be after 'To Date'")

# ---------------- HELPER FUNCTIONS ----------------

def detect_exchange(ticker):
    """
    Determine if ticker is a commodity mapping, Indian ticker or US ticker.
    Returns a ticker that yfinance can use.
    """
    t_up = ticker.upper()
    commodity_map = {'GOLD1': 'GOLDBEES.NS', 'GOLD': 'GOLDBEES.NS', 'GOLDINR': 'GOLDBEES.NS',
                     'GOLDBEES': 'GOLDBEES.NS', 'SILVERETF': 'SILVERBEES.NS', 'SILVER': 'SILVERBEES.NS'}
    if t_up in commodity_map:
        return commodity_map[t_up]
    # If user already included an exchange suffix or short tickers
    if ticker.endswith('.NS') or ticker.endswith('.BO') or '.' in ticker or len(ticker) <= 2:
        return ticker
    # Some common US tickers (quick mapping)
    us_tickers = {'AAPL','MSFT','GOOGL','GOOG','AMZN','TSLA','META','FB','NVDA','AMD','INTC','PYPL','SQ','CRM','NFLX','UBER','LYFT','SPOT','ZM','DBX','SNAP','TWTR','PINS','ROKU','ABNB','DPZ','DASH','COIN','MSTR','RIOT','MARA'}
    if t_up in us_tickers:
        return ticker
    # Try appending suffixes and testing download to infer exchange
    for suffix in ['.NS', '.BO']:
        try:
            with SuppressYFinanceErrors():
                data = yf.download(f"{ticker}{suffix}", period='1d', progress=False, timeout=5)
                if data is not None and not data.empty:
                    return f"{ticker}{suffix}"
        except:
            continue
    # default to Indian NSE ticker
    return f"{ticker}.NS"

# Apply detect_exchange to user tickers while preserving original behavior
tickers = [detect_exchange(t)[0] if isinstance(detect_exchange(t), tuple) else detect_exchange(t) for t in tickers]

def is_indian_stock(ticker):
    """Return True if the ticker appears to be listed on Indian exchanges."""
    return ticker.endswith(".NS") or ticker.endswith(".BO")

def create_sample_data(ticker):
    """
    Generate two years of synthetic daily price data when real data cannot be fetched.
    This keeps UI functional when downloads fail.
    """
    dates = pd.date_range(end=datetime.now(), periods=730, freq='D')
    np.random.seed(hash(ticker) % 2**32)
    returns = np.random.normal(0.0005, 0.02, 730)
    prices = 100 * np.exp(np.cumsum(returns))
    hist = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.01, 730)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.02, 730))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, 730))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 100000000, 730),
    })
    hist['Datetime'] = pd.to_datetime(hist['Date'])
    return hist

@st.cache_data(show_spinner=False)
def fetch_historical_data(ticker):
    """
    Robust historical data fetcher. It tries yfinance first, and if that fails
    it returns synthetic data via create_sample_data.
    - Converts MultiIndex columns if present.
    - Ensures numeric columns are coerced to numeric.
    - Adds a 'Datetime' column from the index for downstream usage.
    """
    try:
        with SuppressYFinanceErrors():
            hist = yf.download(ticker, period="3y", interval="1d", progress=False, timeout=10)
        if (hist is None or hist.empty) and is_indian_stock(ticker):
            # try alternate Indian exchange suffix
            alt_suffix = ".BO" if ticker.endswith(".NS") else ".NS"
            alt_ticker = ticker.rsplit(".",1)[0] + alt_suffix
            with SuppressYFinanceErrors():
                hist = yf.download(alt_ticker, period="3y", interval="1d", progress=False, timeout=10)
        if hist is None or hist.empty:
            return create_sample_data(ticker)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [col[0] for col in hist.columns]
        hist['Datetime'] = pd.to_datetime(hist.index)
        for col in ["Open","High","Low","Close","Volume"]:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors='coerce')
        hist.dropna(subset=["Close"], inplace=True)
        return hist.sort_values("Datetime")
    except:
        return create_sample_data(ticker)

def fetch_live_price_fh(ticker):
    """
    Fetch live quote from Finnhub (if available).
    Returns a dict with expected keys or None if failure.
    """
    symbol = ticker.split(".")[0] if "." in ticker else ticker
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        if "c" not in res or res["c"]==0 or "error" in res:
            return None
        change = res["c"] - res.get("pc",0)
        pct_change = (change/res.get("pc",1))*100 if res.get("pc") else 0
        return {
            "Open": res.get("o",0),
            "High": res.get("h",0),
            "Low": res.get("l",0),
            "Close": res["c"],
            "Prev Close": res.get("pc",0),
            "Change": change,
            "% Change": pct_change,
            "Volume": res.get("v",0),
            "Time": datetime.now().strftime("%H:%M:%S")
        }
    except:
        return None

def fetch_live_price_yf(ticker, intraday=True):
    """
    Attempt to fetch live price using yfinance Ticker.history.
    Tries small intraday intervals first, then falls back to daily.
    """
    with SuppressYFinanceErrors():
        tkr = yf.Ticker(ticker)
    periods = [("1d","1m"),("5d","5m")] if intraday else [("5d","1d"),("1mo","1d")]
    for period, interval in periods:
        try:
            with SuppressYFinanceErrors():
                data = tkr.history(period=period, interval=interval, progress=False, timeout=10)
            if data is not None and not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data)>1 else latest
                change = latest["Close"]-prev["Close"]
                return {
                    "Open": latest.get("Open",latest["Close"]),
                    "High": latest.get("High",latest["Close"]),
                    "Low": latest.get("Low",latest["Close"]),
                    "Close": latest["Close"],
                    "Prev Close": prev["Close"],
                    "Change": change,
                    "% Change": (change/prev["Close"]*100) if prev["Close"] else 0,
                    "Volume": int(latest.get("Volume",0)),
                    "Time": datetime.now().strftime("%H:%M:%S")
                }
        except:
            continue
    return None

def create_live_from_historical(hist):
    """
    Create a fake 'live' price snapshot using the latest historical row.
    This acts as a fallback when no real-time API is available.
    """
    if hist is None or hist.empty:
        return None
    latest, prev = hist.iloc[-1], hist.iloc[-2] if len(hist)>1 else hist.iloc[-1]
    change = latest["Close"] - prev["Close"]
    return {
        "Open": latest.get("Open", latest["Close"]),
        "High": latest.get("High", latest["Close"]),
        "Low": latest.get("Low", latest["Close"]),
        "Close": latest["Close"],
        "Prev Close": prev["Close"],
        "Change": change,
        "% Change": (change/prev["Close"]*100) if prev["Close"] else 0,
        "Volume": int(latest.get("Volume",0)),
        "Time": datetime.now().strftime("%H:%M:%S")
    }

# ---------------- DASHBOARD TITLE ----------------
st.title("📈 Stock Price Prediction Using Time Series Analysis")
# Part 2 of 2
# Continue the file from Part 1.

# ================= Part 2 =================
# ---------------- HELPERS FOR MODELING ----------------

def color_values(val):
    """
    Colorize positive values green, negative red for DataFrame styling.
    """
    if isinstance(val, (int, float, np.floating, np.integer)):
        if val > 0: return f'color: {theme["up"]}; font-weight: bold'
        elif val < 0: return f'color: {theme["down"]}; font-weight: bold'
    return f'color: {theme["text"]}'

@st.cache_resource(show_spinner=False)
def train_arima_model(series):
    """
    Train an ARIMA model on the provided close price series.
    Keeps parameters fixed to (5,1,0) as in original script.
    """
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    return model_fit

@st.cache_resource(show_spinner=False)
def train_lstm_model(series, look_back=10, epochs=10):
    """
    Train a basic 2-layer LSTM on the provided price series.
    Returns (model, scaler, X) where X is the transformed input sequences.
    """
    if len(series) <= look_back:
        return None, None, None
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(np.array(series).reshape(-1,1))
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i,0])
        y.append(data_scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    if X.size == 0:
        return None, None, None
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    return model, scaler, X

def generate_arima_prediction(model_fit, last_date, periods=10, include_today=False):
    """
    Generate future ARIMA predictions and return as DataFrame with Date & ARIMA_Prediction columns.
    Note: last_date is a pandas Timestamp (date). By default predictions start next day (include_today toggles).
    """
    try:
        forecast = model_fit.forecast(steps=periods)
    except:
        forecast = [np.nan]*periods
    start_date = last_date if include_today else last_date + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=periods)
    return pd.DataFrame({"Date": dates, "ARIMA_Prediction": forecast})

def generate_lstm_prediction(_model_data, last_date, live_price=None, look_back=10, periods=10, include_today=False):
    """
    Generate future LSTM predictions. Accepts the trained (model, scaler, X) tuple.
    If live_price is provided, it replaces the last value in the LSTM input sequence for a 'hot-start'.
    Returns DataFrame with Date & LSTM_Prediction columns.
    """
    model, scaler, X = _model_data
    preds = []
    if model is not None and X is not None and len(X)>0:
        last_seq = X[-1].copy()
        if live_price is not None:
            # replace last element of sequence with scaled live price for better short-term prediction
            try:
                last_seq[-1] = scaler.transform(np.array([[live_price]]))[0,0]
            except:
                pass
        for _ in range(periods):
            pred = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)
            preds.append(pred[0,0])
            last_seq = np.append(last_seq[1:], pred, axis=0)
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    else:
        preds = [np.nan]*periods
    start_date = last_date if include_today else last_date + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=periods)
    return pd.DataFrame({"Date": dates, "LSTM_Prediction": preds})

def compute_metrics(y_true, y_pred):
    """
    Compute RMSE and MAE between y_true and y_pred, ignoring NaNs in y_pred.
    """
    mask = ~np.isnan(y_pred)
    y_true, y_pred = np.array(y_true)[mask], np.array(y_pred)[mask]
    if len(y_pred)==0 or len(y_true)==0:
        return np.nan, np.nan
    return sqrt(mean_squared_error(y_true,y_pred)), mean_absolute_error(y_true,y_pred)

def recommendation_box(ticker, live, hist, forecast_df=None):
    """
    Build an HTML card with a simple moving-average based trend and buy/sell levels.
    Returns an HTML string to be rendered with unsafe_allow_html=True.
    """
    try:
        price = live["Close"]
        forecast_price = forecast_df["Combined_Prediction"].iloc[0] if forecast_df is not None and not forecast_df.empty else price
        if hist is None or len(hist)<20:
            sma_20 = sma_50 = price
            trend = "Neutral"
        else:
            sma_20_vals = hist["Close"].rolling(20).mean().dropna()
            sma_50_vals = hist["Close"].rolling(50).mean().dropna()
            sma_20 = sma_20_vals.iloc[-1] if len(sma_20_vals)>0 else price
            sma_50 = sma_50_vals.iloc[-1] if len(sma_50_vals)>0 else price
            trend = "Bullish" if sma_20 > sma_50 else "Bearish"
        buy_level, sell_level = price*1.02, price*0.98
        risk_buy = abs((price-buy_level)/buy_level*100)
        risk_sell = abs((sell_level-price)/sell_level*100)
    except:
        price = forecast_price = live["Close"]
        trend, buy_level, sell_level, risk_buy, risk_sell = "Neutral", price*1.02, price*0.98, 2.0, 2.0
    risk_label = lambda r: ("Good","#22C55E") if r<1 else (("Low Risk","#86efac") if r<3 else (("Risk","#facc15") if r<5 else ("High Risk","#ef4444")))

    label_buy, color_buy = risk_label(risk_buy)
    label_sell, color_sell = risk_label(risk_sell)
    return f"""<div style='background:{theme["card"]};padding:15px;border-radius:12px;font-family:monospace;box-shadow:0px 0px 10px #000;' >
        <b style='color:{theme["accent"]};font-size:18px'>Recommendation for {ticker}</b><br>
        <b>Trend:</b> {trend}<br><b style='color:{theme["text"]}'>Forecast Today Price:</b> {forecast_price:.2f}<br><br>
        <b style='color:{theme["up"]}'>Buy Above:</b> {buy_level:.2f} | Stoploss: {price*0.98:.2f}<br>
        <b style='color:{color_buy}'>Risk: {risk_buy:.2f}% ({label_buy})</b><br><hr>
        <b style='color:{theme["down"]}'>Sell Below:</b> {sell_level:.2f} | Stoploss: {price*1.02:.2f}<br>
        <b style='color:{color_sell}'>Risk: {risk_sell:.2f}% ({label_sell})</b></div>"""

# ---------------- INITIALIZE DASHBOARD PLACEHOLDERS ----------------
tabs = st.tabs(tickers)
placeholders = {k:{} for k in ['chart','live','rec','table','prediction','prediction_table','forecast_chart','live_price']}
live_data_x, live_data_y, hist_data, arima_models, lstm_models = {}, {}, {}, {}, {}

for i, ticker in enumerate(tickers):
    try:
        with tabs[i]:
            st.subheader(f"🔹 {ticker}")
            # Sections to be displayed for each ticker
            sections = [("### 💹 Current Price","live_price"),
                        ("### 💰 Live Price Table","live"),
                        ("### 📉 Recommendation Box","rec"),
                        ("### 📈 Historical Price Chart","chart"),
                        ("### 📊 Historical Data Table","table"),
                        ("### 🧮 Model Forecast Chart","prediction"),
                        ("### 📋 Model Forecast Table","prediction_table"),
                        ("### 📊 Model Evaluation Metrics","forecast_chart")]
            for header, key in sections:
                st.markdown(header)
                # store placeholders for each ticker for subsequent updates
                placeholders[key][ticker] = st.empty() if key!="forecast_chart" else {"EVAL": st.empty()}

            # Fetch historical data (cached function)
            hist = fetch_historical_data(ticker)
            if hist is None or hist.empty:
                st.warning(f"⚠️ Could not fetch data for {ticker}.")
                continue
            hist_data[ticker] = hist

            # Train models only if sufficient data
            if len(hist) > 50:
                arima_models[ticker] = train_arima_model(hist["Close"])
                lstm_models[ticker] = train_lstm_model(hist["Close"])

            # Initialize chart (historical + placeholder for live)
            hist_filtered = hist[(hist["Datetime"].dt.date >= from_date) & (hist["Datetime"].dt.date <= to_date)].copy()
            if hist_filtered.empty:
                hist_filtered = hist.tail(30).copy()
            hist_filtered = hist_filtered.sort_values("Datetime")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_filtered["Datetime"], y=hist_filtered["Close"], mode="lines",
                                     line=dict(color=theme["accent"], width=2), name="Historical Price"))
            # live price trace placeholder (starts empty)
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers",
                                     line=dict(color=theme["up"], width=3), marker=dict(size=6), name="Live Price"))
            # Volume bar (if available)
            if len(hist_filtered)>1 and "Volume" in hist_filtered.columns:
                vol_colors = [theme["up"] if hist_filtered["Close"].iloc[i]>=hist_filtered["Close"].iloc[i-1] else theme["down"] for i in range(1,len(hist_filtered))]
                fig.add_trace(go.Bar(x=hist_filtered["Datetime"].iloc[1:], y=hist_filtered["Volume"].iloc[1:], marker_color=vol_colors, opacity=0.3, name="Volume", yaxis="y2"))
            fig.update_layout(paper_bgcolor=theme["background"], plot_bgcolor=theme["background"], font=dict(color=theme["text"]),
                              xaxis=dict(showgrid=True, gridcolor="#1F2937", rangeslider=dict(visible=True)),
                              yaxis=dict(title="Price", showgrid=True, gridcolor="#1F2937", side="right"),
                              yaxis2=dict(overlaying="y", side="left", showgrid=False, title="Volume"),
                              hovermode="x unified", dragmode="zoom", height=600, margin=dict(l=20,r=20,t=50,b=50), showlegend=True)
            # store the figure in session state for live updates
            st.session_state[f"fig_live_{ticker}"] = fig
            placeholders['chart'][ticker].plotly_chart(fig, use_container_width=True, key=f"{ticker}_init")
            live_data_x[ticker], live_data_y[ticker] = [], []

    except Exception as e:
        st.warning(f"⚠️ Error initializing {ticker}: {str(e)[:50]}")
        continue

# ================= Part 3 =================
# ---------------- LIVE UPDATE LOOP ----------------
# NOTE: This loop will run indefinitely in Streamlit until the app stops.
#       It updates live prices, tables, forecast and charts on each tick.
while True:
    for ticker in tickers:
        # Skip if ticker was not initialized
        if ticker not in live_data_x or ticker not in hist_data:
            continue

        try:
            # Re-fetch historical data (cached so not heavy)
            hist = fetch_historical_data(ticker)
            if hist is None or hist.empty or len(hist) < 10:
                continue

            # Fetch live price using prioritized methods
            if is_indian_stock(ticker):
                live = fetch_live_price_yf(ticker, intraday=False) or create_live_from_historical(hist)
            else:
                live = fetch_live_price_fh(ticker) or fetch_live_price_yf(ticker, intraday=True) or create_live_from_historical(hist)

            if not live:
                continue

            # Ensure live dict has required keys (defensive)
            required_keys = ["Close","Open","High","Low","Prev Close","Change","% Change","Volume","Time"]
            if not all(k in live for k in required_keys):
                continue

        except Exception:
            # If anything goes wrong fetching live data, skip this ticker for this cycle
            continue

        # ---------- Live Price Card ----------
        pct = float(live.get('% Change',0.0))
        if pct > 0.1:
            icon, trend_label, trend_color = '▲','Rising',theme['up']
        elif pct < -0.1:
            icon, trend_label, trend_color = '▼','Falling',theme['down']
        else:
            icon, trend_label, trend_color = '▬','Stable',theme['text']
        currency_symbol = "₹" if is_indian_stock(ticker) else "USD"

        live_price_html = (
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"<div style='font-size:25px;color:{theme['accent']};font-weight:700;'>"
            f"{currency_symbol} {live['Close']:.2f}</div>"
            f"<div style='font-size:18px;color:{trend_color};font-weight:700;'>"
            f"{icon} {pct:+.2f}%</div>"
            f"<div style='background:{theme['card']};color:{trend_color};padding:6px 10px;border-radius:8px;font-size:13px;'>"
            f"{trend_label}</div>"
            f"</div>"
        )
        # render the live price card
        placeholders['live_price'][ticker].markdown(live_price_html, unsafe_allow_html=True)

        # ---------- Live Price Table ----------
        try:
            live_data_df = pd.DataFrame({
                "Open":[f"{live['Open']:.2f}"],
                "High":[f"{live['High']:.2f}"],
                "Low":[f"{live['Low']:.2f}"],
                "Close":[f"{live['Close']:.2f}"],
                "Prev Close":[f"{live['Prev Close']:.2f}"],
                "Change":[f"{live['Change']:+.2f}"],
                "% Change":[f"{live['% Change']:+.2f}%"],
                "Volume":[f"{int(live['Volume']):,}"],
                "Time":[live['Time']]
            })
            styled_live = live_data_df.style.map(color_values, subset=["Change","% Change"])
            placeholders['live'][ticker].dataframe(styled_live, use_container_width=True, key=f"live_table_{ticker}_{time.time()}")
        except Exception as e:
            placeholders['live'][ticker].warning(f"Could not display live table: {str(e)[:30]}")

        # ---------- Forecast Predictions ----------
        # Generate predictions only if models exist for the ticker
        if ticker in arima_models and lstm_models.get(ticker):
            last_date = pd.Timestamp(datetime.now().date())

            # generate ARIMA & LSTM prediction DataFrames
            arima_df = generate_arima_prediction(arima_models[ticker], last_date, periods=forecast_days, include_today=True)
            # IMPORTANT: convert the Date column to plain date (no time) for display in tables
            arima_df["Date"] = arima_df["Date"].dt.date

            lstm_df = generate_lstm_prediction(lstm_models[ticker], last_date, live_price=live["Close"], periods=forecast_days, include_today=True)
            # IMPORTANT: convert the Date column to plain date (no time) for display in tables
            lstm_df["Date"] = lstm_df["Date"].dt.date

            # merge on Date (now both are date objects)
            df_preds = pd.merge(arima_df, lstm_df, on="Date")
            # ensure merged Date is date-only (defensive)
            df_preds["Date"] = pd.to_datetime(df_preds["Date"]).dt.date
            df_preds["Combined_Prediction"] = (df_preds["ARIMA_Prediction"] + df_preds["LSTM_Prediction"])/2
        else:
            df_preds = None

        # ---------- Recommendation Box ----------
        try:
            placeholders['rec'][ticker].markdown(recommendation_box(ticker, live, hist, df_preds), unsafe_allow_html=True)
        except Exception as e:
            placeholders['rec'][ticker].error(f"⚠️ Could not generate recommendation: {str(e)[:50]}")

        # ---------- Update Historical Chart ----------
        if ticker in live_data_x and ticker in live_data_y and f"fig_live_{ticker}" in st.session_state and len(st.session_state[f"fig_live_{ticker}"].data)>=2:
            fig = st.session_state[f"fig_live_{ticker}"]
            hist_filtered = hist[(hist["Datetime"].dt.date>=from_date) & (hist["Datetime"].dt.date<=to_date)].copy()
            if hist_filtered.empty:
                hist_filtered = hist.tail(30).copy()
            hist_filtered = hist_filtered.sort_values("Datetime")
            # update historical trace (data[0])
            fig.data[0].x = hist_filtered["Datetime"].tolist()
            fig.data[0].y = hist_filtered["Close"].tolist()
            # Update volume bars if present
            if len(fig.data)>2 and "Volume" in hist_filtered.columns and len(hist_filtered)>1:
                vol_colors = [theme["up"] if hist_filtered["Close"].iloc[i]>=hist_filtered["Close"].iloc[i-1] else theme["down"] for i in range(1,len(hist_filtered))]
                fig.data[2].x = hist_filtered["Datetime"].iloc[1:].tolist()
                fig.data[2].y = hist_filtered["Volume"].iloc[1:].tolist()
                fig.data[2].marker.color = vol_colors

        # Append live price to local arrays for live trace
        live_data_x[ticker].append(datetime.now())
        live_data_y[ticker].append(live["Close"])
        if len(live_data_x[ticker])>500:
            live_data_x[ticker] = live_data_x[ticker][-500:]
            live_data_y[ticker] = live_data_y[ticker][-500:]
        # Update live trace appearance and push chart to UI
        if f"fig_live_{ticker}" in st.session_state:
            fig = st.session_state[f"fig_live_{ticker}"]
            fig.data[1].x = [pd.Timestamp(dt) for dt in live_data_x[ticker]]
            fig.data[1].y = live_data_y[ticker]
            fig.data[1].line.color = theme["up"] if len(live_data_y[ticker])<2 or live_data_y[ticker][-1]>=live_data_y[ticker][-2] else theme["down"]
            placeholders['chart'][ticker].plotly_chart(fig, use_container_width=True, key=f"{ticker}_live_chart_{time.time()}")

                            # ---------- Update Historical Table ----------
        try:
            # Filter by date range
            hist_filtered = hist[
                (hist["Datetime"].dt.date >= from_date) &
                (hist["Datetime"].dt.date <= to_date)
            ].copy()

            if hist_filtered.empty:
                hist_filtered = hist.tail(30).copy()

            # Create Date column (remove time)
            hist_filtered["Date"] = hist_filtered["Datetime"].dt.strftime("%Y-%m-%d")

            # Add % Change column
            hist_filtered["% Change"] = hist_filtered["Close"].pct_change() * 100

            # Build display table
            hist_display = hist_filtered.sort_values("Datetime", ascending=False)[
                ["Date", "Open", "High", "Low", "Close", "Volume", "% Change"]
            ].copy()

            # Reset index so styling works correctly
            hist_display.reset_index(drop=True, inplace=True)

            # --- Color only Close & % Change columns ---
            def highlight_close(val):
                """Color green for up, red for down, white for missing."""
                if pd.isna(val):
                    return "color: inherit;"
                return "color: #10B981;" if val >= 0 else "color: #EF4444;"

            def highlight_pct(val):
                """Same coloring for % Change."""
                if pd.isna(val):
                    return "color: inherit;"
                return "color: #10B981;" if val >= 0 else "color: #EF4444;"

            # Apply styles
            hist_display_styled = (
                hist_display.style
                .applymap(highlight_close, subset=["Close"])
                .applymap(highlight_pct, subset=["% Change"])
                .set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#374151"),
                        ("color", "#F3F4F6"),
                        ("font-weight", "bold"),
                        ("padding", "6px 8px"),
                        ("border-color", "#4B5563"),
                    ]},
                    {"selector": "td", "props": [
                        ("padding", "6px 8px"),
                        ("border-color", "#1F2937"),
                    ]},
                ])
            )

            # Display the table
            placeholders['table'][ticker].dataframe(
                hist_display_styled,
                use_container_width=True
            )

        except Exception as e:
            placeholders['table'][ticker].warning(f"Historical Table Error: {str(e)}")


        # ---------- Forecast Table & Charts ----------
        if df_preds is not None:
            try:
                # Display the forecast table with Date column as plain date (no time)
                # Format numeric columns and color them
                styled_forecast = df_preds.style.format({"ARIMA_Prediction":"{:.2f}","LSTM_Prediction":"{:.2f}","Combined_Prediction":"{:.2f}"})
                styled_forecast = styled_forecast.map(color_values, subset=["ARIMA_Prediction","LSTM_Prediction","Combined_Prediction"])
                placeholders['prediction_table'][ticker].dataframe(
                    styled_forecast,
                    use_container_width=True, key=f"forecast_table_{ticker}_{time.time()}"
                )

                # Forecast charts (ARIMA, LSTM, Combined)
                # Use original numeric training values for plotting historical series.
                train_values, train_dates = hist["Close"].values, hist["Datetime"]
                forecast_tabs = placeholders['prediction'][ticker].tabs(["ARIMA","LSTM","ARIMA + LSTM"])
                chart_configs = [
                    (arima_df["Date"], arima_df["ARIMA_Prediction"], "ARIMA Forecast", "#FFA500", True),
                    (lstm_df["Date"], lstm_df["LSTM_Prediction"], "LSTM Forecast", "#FF4500", True),
                    (df_preds["Date"], df_preds["Combined_Prediction"], "Combined Forecast", "#00FFFF", False)
                ]
                # Note: for chart x axes we prefer datetime-like values. For ARIMA/LSTM dates we've converted to date objects for table display,
                # but Plotly will accept these date objects for plotting as well.
                for idx,(dates,values,name,color,show_hist) in enumerate(chart_configs):
                    with forecast_tabs[idx]:
                        fig_forecast = go.Figure()
                        if show_hist:
                            fig_forecast.add_trace(go.Scatter(x=train_dates, y=train_values, mode="lines", name="Historical", line=dict(color="#22C55E")))
                        fig_forecast.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers", name=name, line=dict(color=color,dash="dash")))
                        fig_forecast.update_layout(title=f"{name} Chart", paper_bgcolor=theme["background"], font=dict(color=theme["text"]), hovermode="x unified")
                        st.plotly_chart(fig_forecast, use_container_width=True, key=f"{name.lower().replace(' ','_')}_chart_{ticker}_{time.time()}")

                # Evaluation metrics computed against the latest available historical values
                last_test_values = hist["Close"].iloc[-forecast_days:].values if len(hist)>forecast_days else hist["Close"].values
                arima_rmse, arima_mae = compute_metrics(last_test_values, df_preds["ARIMA_Prediction"].values[:len(last_test_values)])
                lstm_rmse, lstm_mae = compute_metrics(last_test_values, df_preds["LSTM_Prediction"].values[:len(last_test_values)])
                combined_rmse, combined_mae = compute_metrics(last_test_values, df_preds["Combined_Prediction"].values[:len(last_test_values)])
                metrics_df = pd.DataFrame({
                    "Model":["ARIMA","LSTM","Combined"],
                    "RMSE":[arima_rmse,lstm_rmse,combined_rmse],
                    "MAE":[arima_mae,lstm_mae,combined_mae]
                })
                # Plot metrics
                eval_fig = go.Figure(data=[
                    go.Bar(name='RMSE', x=metrics_df["Model"], y=metrics_df["RMSE"], marker_color='#FFA500'),
                    go.Bar(name='MAE', x=metrics_df["Model"], y=metrics_df["MAE"], marker_color='#00FFFF')
                ])
                eval_fig.update_layout(barmode='group', paper_bgcolor=theme["background"], font=dict(color=theme["text"]), hovermode="x unified", margin=dict(l=20,r=20,t=30,b=20))
                metrics_container = placeholders['forecast_chart'][ticker]["EVAL"].container()
                metrics_container.plotly_chart(eval_fig, use_container_width=True, key=f"eval_chart_{ticker}_{time.time()}")
                metrics_container.dataframe(metrics_df.style.format({"RMSE":"{:.4f}","MAE":"{:.4f}"}), use_container_width=True, key=f"eval_table_{ticker}_{time.time()}")

            except Exception:
                # ignore forecasting display errors to avoid breaking the live loop
                pass

    # Sleep for the configured refresh rate before updating again
    time.sleep(refresh_rate)
