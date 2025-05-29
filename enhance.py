import os
import joblib
import requests
import pandas as pd
import numpy as np
import torch

# Scaling
from sklearn.preprocessing import MinMaxScaler

# Plotting and Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Reinforcement Learning Libraries
import gym
from gym import spaces
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

# Streamlit for interactive UI
import streamlit as st

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, TimeDistributed, Flatten, LSTM, GRU, concatenate

# Sentiment Analysis
from transformers import pipeline  # Import pipeline from transformers

# --------------------------------------------------------------------
# Helper: always load SB3 models onto CPU-only runtimes
# --------------------------------------------------------------------
def safe_load(path, algo_cls):
    """
    Load a Stable-Baselines3 model on CPU.

    Forces device='cpu' so CUDA-trained checkpoints do not crash
    on Streamlit Cloud’s CPU build of PyTorch.
    """
    try:
        return algo_cls.load(path, device='cpu')
    except FileNotFoundError:
        st.warning(f"Model file not found: {path}")
    except Exception as e:
        st.error(f"Could not load {path}: {e}")
    return None

# =============================================================================
# 1) Data Fetching
# =============================================================================


def fetch_live_data(tickers, retries=3):
    """
    Fetch 15-minute historical data from FinancialModelingPrep.
    Requires FMP_API_KEY to be set in your environment or input via Streamlit.
    """
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        st.error("API key not found. Please set FMP_API_KEY in your environment.")
        api_key = st.text_input("Enter your FinancialModelingPrep API Key:")
        if not api_key:
            st.error("API key not provided. Cannot fetch live data.")
            return data

    for ticker in tickers:
        for attempt in range(retries):
            try:
                ticker_api = ticker.replace('/', '')
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/15min/{ticker_api}?apikey={api_key}'
                response = requests.get(url)
                response.raise_for_status()
                data_json = response.json()

                if not data_json or len(data_json) < 1:
                    continue  # Retry if empty

                df = pd.DataFrame(data_json)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                df.rename(columns={
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume'
                }, inplace=True)

                # Sort by ascending datetime
                df.sort_index(inplace=True)

                data[ticker] = df
                break  # Break if we successfully fetched data
            except Exception as e:
                if attempt < retries - 1:
                    continue
                else:
                    st.error(f"Error fetching data for {ticker}: {e}")
                    break

    return data

# =============================================================================
# 2) Fetch News Data
# =============================================================================


def fetch_news_data():
    """
    Fetch news data from NewsAPI and economic data from FRED API.
    Requires NEWS_API_KEY_NEWSAPI and NEWS_API_KEY_FRED to be set in your environment or input via Streamlit.
    """
    news_api_key = os.getenv("NEWS_API_KEY_NEWSAPI")
    fred_api_key = os.getenv("NEWS_API_KEY_FRED")

    if not news_api_key:
        st.error(
            "NewsAPI key not found. Please set NEWS_API_KEY_NEWSAPI in your environment.")
        news_api_key = st.text_input("Enter your NewsAPI Key:")
        if not news_api_key:
            st.error("NewsAPI key not provided. Cannot fetch news data.")
            return [], 0
    if not fred_api_key:
        st.error(
            "FRED API key not found. Please set NEWS_API_KEY_FRED in your environment.")
        fred_api_key = st.text_input("Enter your FRED API Key:")
        if not fred_api_key:
            st.error("FRED API key not provided. Cannot fetch economic data.")
            return [], 0

    # Fetch headlines from NewsAPI
    news_url = f'https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={news_api_key}'

    news_response = requests.get(news_url)
    news_data = news_response.json()

    # Extract headlines
    headlines = [article['title'] for article in news_data.get('articles', [])]

    # Fetch economic data from FRED API (e.g., interest rates)
    fred_series_id = 'FEDFUNDS'  # Effective Federal Funds Rate
    fred_url = f'https://api.stlouisfed.org/fred/series/observations?series_id={fred_series_id}&api_key={fred_api_key}&file_type=json'

    fred_response = requests.get(fred_url)
    fred_data = fred_response.json()

    try:
        # Extract recent economic indicator
        fred_observations = fred_data.get('observations', [])
        latest_fred_value = float(
            fred_observations[-1]['value']) if fred_observations else 0
    except (KeyError, IndexError, ValueError):
        latest_fred_value = 0

    return headlines, latest_fred_value

# =============================================================================
# 3) Sentiment Analysis of News Headlines
# =============================================================================
def compute_sentiment_score(headlines):
    """
    Compute average sentiment polarity of the news headlines using a lightweight BERT transformer.
    Compatible with Streamlit Cloud by forcing the PyTorch backend.
    """
    if not headlines:
        return 0  # Neutral sentiment if no headlines

    try:
        sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",  # Use PyTorch to ensure compatibility
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.warning("⚠️ Sentiment model could not be loaded. Setting neutral sentiment.")
        print(f"[ERROR] Sentiment model loading failed: {e}")
        return 0

    sentiment_scores = []

    for headline in headlines:
        try:
            result = sentiment_analyzer(headline)[0]
            if result['label'] == 'POSITIVE':
                sentiment_scores.append(result['score'])
            elif result['label'] == 'NEGATIVE':
                sentiment_scores.append(-result['score'])
            else:
                sentiment_scores.append(0)
        except Exception as e:
            st.warning(f"Error analyzing sentiment for headline: '{headline}'. Skipping.")
            sentiment_scores.append(0)

    return np.mean(sentiment_scores) if sentiment_scores else 0

# =============================================================================
# 4) Indicator Computations (Advanced Feature Engineering)
# =============================================================================


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line


def compute_bollinger_band_width(df, period=20, num_std=2):
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    df['BB_Std'] = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + num_std * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - num_std * df['BB_Std']
    df['BB_Bandwidth'] = (
        (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
    return df


def compute_OBV(df):
    """Compute On-Balance Volume."""
    df['Delta_Close'] = df['Close'].diff()
    df['Direction'] = np.where(
        df['Delta_Close'] > 0, 1, np.where(df['Delta_Close'] < 0, -1, 0))
    df['Volume_Adjusted'] = df['Volume'] * df['Direction']
    df['OBV'] = df['Volume_Adjusted'].cumsum()
    return df


def compute_ATR(df, period=14):
    """Compute Average True Range."""
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR'], inplace=True)
    return df


def compute_EMAs(df, periods=[21, 50]):
    """Compute Exponential Moving Averages."""
    for period in periods:
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df


def compute_ADX(df, period=14):
    """
    Compute the Average Directional Index (ADX).
    """
    df = df.copy()

    # True Range (TR)
    df['TR'] = df[['High', 'Close']].shift(1).max(
        axis=1) - df[['Low', 'Close']].shift(1).min(axis=1)

    # Directional Movement
    df['+DM'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        df['High'] - df['High'].shift(1),
        0.0
    )
    df['+DM'] = df['+DM'].clip(lower=0)
    df['-DM'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        df['Low'].shift(1) - df['Low'],
        0.0
    )
    df['-DM'] = df['-DM'].clip(lower=0)

    # Smoothed averages
    df['TR14'] = df['TR'].rolling(window=period).sum()
    df['+DM14'] = df['+DM'].rolling(window=period).sum()
    df['-DM14'] = df['-DM'].rolling(window=period).sum()

    # Directional Indicators
    df['+DI14'] = (df['+DM14'] / df['TR14']) * 100
    df['-DI14'] = (df['-DM14'] / df['TR14']) * 100

    # ADX computation
    df['DX'] = (abs(df['+DI14'] - df['-DI14']) /
                (df['+DI14'] + df['-DI14'])) * 100
    df['ADX'] = df['DX'].rolling(window=period).mean()

    df.drop(columns=['TR', '+DM', '-DM', 'TR14', '+DM14',
            '-DM14', '+DI14', '-DI14', 'DX'], inplace=True)
    df.dropna(inplace=True)
    return df

# =============================================================================
# 5) Add Transformer Predictions (Placeholder)
# =============================================================================


def add_transformer_predictions(df):
    """
    Adds a placeholder 'Transformer_Prediction' column to the DataFrame.
    This is necessary to match the observation space used during training.
    """
    df['Transformer_Prediction'] = 0  # Placeholder value
    return df

# =============================================================================
# 6) Enhanced Feature Engineering
# =============================================================================


def enhance_features(df):
    df = df.copy()

    # RSI
    df['RSI'] = compute_RSI(df['Close'])

    # MACD
    df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])

    # Bollinger Band Width
    df = compute_bollinger_band_width(df)

    # OBV
    df = compute_OBV(df)

    # ATR
    df = compute_ATR(df)

    # EMAs
    df = compute_EMAs(df)

    # ADX
    df = compute_ADX(df)

    # Drop unnecessary columns
    df.drop(columns=['Delta_Close', 'Direction', 'Volume_Adjusted'], inplace=True, errors='ignore')
    df.dropna(inplace=True)

    return df

# =============================================================================
# 7) Custom Trading Environment for RL
# =============================================================================


class TradingEnv(gym.Env):
    """A custom trading environment for RL."""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, sentiment_score, economic_indicator, take_profit=0.05, stop_loss=0.05):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.total_steps = len(df) - 1

        # Exclude 'date' column
        num_features = len(self.df.columns) - 1

        num_external_factors = 3  # sentiment_score, economic_indicator, Transformer_Prediction

        total_obs_size = num_features + num_external_factors

        # Actions: Continuous action between -1 and 1 (sell, hold, buy)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observations: Price data, indicators, sentiment, economic data, Transformer's prediction
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        print(f"Observation Space Shape: {self.observation_space.shape}")

        # Initial balance and holdings
        self.initial_balance = 10000  # Starting with $10,000
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.shares_held = 0
        self.last_price = self.df.loc[self.current_step, 'Close']

        # Validate self.last_price
        if np.isnan(self.last_price) or self.last_price <= 0:
            st.warning(
                "Initial last_price is invalid. Setting to a small positive value.")
            self.last_price = 1e-6  # Small positive value

        # External factors
        self.sentiment_score = sentiment_score
        self.economic_indicator = economic_indicator

        # Trade logging
        self.trade_history = []

        # Risk management parameters
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.shares_held = 0
        self.trade_history = []

        # Ensure the initial price is valid
        self.last_price = self.df.loc[self.current_step, 'Close']
        if np.isnan(self.last_price) or self.last_price <= 0:
            st.warning(
                "Warning: Initial price is invalid. Setting to a small positive value.")
            self.last_price = 1e-6  # Small positive value to avoid division by zero

        return self._next_observation()

    def _next_observation(self):
        # Extract the current row of data
        row = self.df.iloc[self.current_step]

        # Remove 'date' if it's a column
        if 'date' in row.index:
            row = row.drop(['date'])

        # Convert row to list of features
        obs = row.values.tolist()

        # Append external factors: sentiment_score, economic_indicator, Transformer_Prediction
        obs.append(self.sentiment_score)
        obs.append(self.economic_indicator)
        obs.append(self.df.loc[self.current_step, 'Transformer_Prediction'])

        # Convert the list to a numpy array
        obs_array = np.array(obs, dtype='float32')

        # Check for NaN or infinite values in observations
        if np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)):
            st.warning(f"Observation contains invalid values at step {self.current_step}. Replacing with zeros.")
            obs_array = np.nan_to_num(
                obs_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Debug: Print the length of the observation
        print(f"Observation length at step {self.current_step}: {len(obs_array)}")

        return obs_array

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        reward = 0
        done = False

        # Ensure action is scalar
        if isinstance(action, np.ndarray):
            action = action[0]

        # Clip action to the valid range
        action = np.clip(action, -1, 1)

        # Implement position sizing based on action value
        # Positive action: Buy; Negative action: Sell; action magnitude: fraction of balance or holdings
        if action > 0:
            # Buy
            amount_to_invest = self.balance * action  # Fraction of balance to invest
            if amount_to_invest > 0:
                shares_bought = amount_to_invest / current_price
                self.balance -= shares_bought * current_price
                self.shares_held += shares_bought
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'shares': shares_bought,
                    'price': current_price,
                    'total': shares_bought * current_price
                })

        elif action < 0:
            # Sell
            fraction_to_sell = -action  # Fraction of holdings to sell
            shares_to_sell = self.shares_held * fraction_to_sell
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'total': shares_to_sell * current_price
                })

        else:
            # Hold
            pass

        # Update net worth
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price

        # Implement stop-loss
        loss = (self.net_worth - self.initial_balance) / self.initial_balance
        if loss <= -self.stop_loss:
            # Stop-loss triggered: sell all holdings
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'stop_loss_sell',
                    'shares': self.shares_held,
                    'price': current_price,
                    'total': self.shares_held * current_price
                })
                self.shares_held = 0
            done = True  # End episode

        # Immediate reward is the change in net worth
        reward = self.net_worth - prev_net_worth

        # Ensure reward is a valid number
        if np.isnan(reward) or np.isinf(reward):
            st.warning(f"Reward is invalid at step {self.current_step}. Setting reward to zero.")
            reward = 0

        self.last_price = current_price
        self.current_step += 1

        if self.current_step >= self.total_steps:
            done = True

        obs = self._next_observation() if not done else np.zeros(
            self.observation_space.shape)

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        # Print the current balance and net worth
        profit = self.net_worth - self.initial_balance
        st.write(f"Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}")

# =============================================================================
# 8) Prepare Data for RL
# =============================================================================


def prepare_rl_data(df):
    """Scale and prepare data for the RL environment."""
    data = df.copy()
    scaler = MinMaxScaler()

    # Exclude 'date' from features if present
    features = data.drop(columns=['date'], errors='ignore')

    scaled_features = scaler.fit_transform(features)

    # Create DataFrame with scaled features
    data_scaled = pd.DataFrame(scaled_features, columns=features.columns)

    # Add 'date' column back if it was present
    if 'date' in data.columns:
        data_scaled['date'] = data['date'].values
        # Reorder columns to have 'date' first
        cols = ['date'] + [col for col in data_scaled.columns if col != 'date']
        data_scaled = data_scaled[cols]

    # Debug: Print the number of features after scaling
    print(f"Number of features after scaling: {len(data_scaled.columns)}")
    print(f"Columns in data_scaled: {data_scaled.columns.tolist()}")

    return data_scaled, scaler

# =============================================================================
# 9) Performance Monitoring and Visualization
# =============================================================================


def evaluate_model(env, model):
    """Evaluate the RL agent's performance and collect data for visualization."""
    obs = env.reset()
    total_reward = 0
    done = False
    net_worths = []
    steps = []
    actions = []
    prices = []

    while not done:
        action, _states = model.predict(obs)
        # Extract scalar action value
        if isinstance(action, np.ndarray):
            action = action[0]
        else:
            action = float(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        net_worths.append(env.net_worth)
        steps.append(env.current_step)
        actions.append(action)
        prices.append(env.last_price)

    # Create DataFrame for visualization
    results_df = pd.DataFrame({
        'Step': steps,
        'Net Worth': net_worths,
        'Action': actions,
        'Price': prices
    })

    st.write(f"Total Profit: {env.net_worth - env.initial_balance:.2f}")

    return results_df, env.trade_history

# =============================================================================
# 10) Automate Execution (Integration Placeholder)
# =============================================================================


def execute_trades(model, latest_data, sentiment_score, economic_indicator):
    """
    Placeholder function for automating trade execution.
    In a live trading system, this function would interface with a brokerage API.
    """
    # Exclude 'date' from latest_data
    latest_data = latest_data.drop(['date'], errors='ignore')
    obs = latest_data.values.astype('float32')
    # Append external factors: sentiment_score, economic_indicator, Transformer_Prediction
    obs = np.append(obs, [sentiment_score, economic_indicator, latest_data['Transformer_Prediction']])

    # Debug: Print the length of the observation
    print(f"Observation length in execute_trades: {len(obs)}")

    # Check for invalid values in observation
    if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
        st.warning(
            "Observation contains invalid values for trade execution. Replacing with zeros.")
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict action
    action, _states = model.predict(obs)
    # Extract scalar action value
    if isinstance(action, np.ndarray):
        action = action[0]
    else:
        action = float(action)

    # Map action to trade execution
    recommended_action = ""
    if action > 0.01:
        # Buy
        recommended_action = f"Buy {action * 100:.2f}% of available balance"
        st.success(f"Recommended Action: {recommended_action}")
        # Place buy order via brokerage API
    elif action < -0.01:
        # Sell
        recommended_action = f"Sell {abs(action) * 100:.2f}% of holdings"
        st.success(f"Recommended Action: {recommended_action}")
        # Place sell order via brokerage API
    else:
        recommended_action = "Hold"
        st.info("Recommended Action: Hold")
        # Hold - do nothing

    return recommended_action

# =============================================================================
# Helper Functions for Mapping Actions and Decisions
# =============================================================================

def map_action_to_decision_confidence(action_value):
    """
    Map the action value to a decision and confidence level.
    """
    abs_action = abs(action_value)
    if action_value == 0:
        decision = 'Hold'
        confidence = 'Neutral'
    else:
        if action_value > 0:
            decision = 'Buy'
        else:
            decision = 'Sell'
        if abs_action > 0.7:
            confidence = 'High'
        elif abs_action > 0.3:
            confidence = 'Medium'
        else:
            confidence = 'Low'
    return decision, confidence

def get_action_value(model, latest_data, sentiment_score, economic_indicator):
    """
    Get the action value from the model for the latest data.
    """
    # Exclude 'date' from latest_data
    latest_data = latest_data.drop(['date'], errors='ignore')

    # Convert to numpy array
    obs = latest_data.values.astype('float32')

    # Append external factors: sentiment_score, economic_indicator, Transformer_Prediction
    obs = np.append(obs, [sentiment_score, economic_indicator, latest_data['Transformer_Prediction']])

    # Replace NaN or infinite values
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # Debug: Print the length of the observation
    print(f"Observation length in get_action_value: {len(obs)}")

    # Predict action
    action, _states = model.predict(obs)
    # Extract scalar action value
    if isinstance(action, np.ndarray):
        action_value = action[0]
    else:
        action_value = float(action)
    return action_value

# =============================================================================
# 11) Main Execution with Streamlit Interface
# =============================================================================

def main():
    st.title("Enhanced Trading Strategy Application")

    # Sidebar for Inputs
    st.sidebar.header("User Input Parameters")
    default_tickers = ["CC=F", "GC=F", "KC=F", "NG=F", "^GDAXI", "^HSI", "USD/JPY", "ETHUSD", "SOLUSD", "^SPX", "HG=F", "SI=F"]
    tickers = st.sidebar.multiselect(
        "Select Tickers", default_tickers, default=default_tickers)
    stop_loss = st.sidebar.slider("Stop-Loss Threshold (%)", min_value=0.0,
                                  max_value=10.0, value=5.0) / 100  # Convert to fraction
    take_profit = st.sidebar.slider(
        "Take-Profit Threshold (%)", min_value=0.0, max_value=10.0, value=5.0) / 100  # Convert to fraction

    # Fetch external data
    st.sidebar.info("Fetching News Data...")
    headlines, economic_indicator = fetch_news_data()
    sentiment_score = compute_sentiment_score(headlines)
    st.sidebar.write(f"Sentiment Score: {sentiment_score:.2f}")
    st.sidebar.write(
        f"Economic Indicator (FEDFUNDS): {economic_indicator:.2f}%")

    # Add disclaimer
    st.sidebar.markdown("""
    ---
    ### ⚠️ Disclaimer

    This application is provided for **educational and informational purposes only** and does not constitute financial, investment, or trading advice.  
    Any decisions based on the output of this tool are **made at your own risk**.  
    The developer is **not liable** for any losses or damages resulting from use of this application.

    ---
    """)


    # Display headlines
    if st.sidebar.checkbox("Show Latest News Headlines"):
        st.subheader("Latest News Headlines")
        for headline in headlines[:10]:
            st.write(f"- {headline}")

    # Fetch live data
    st.info("Fetching Live Data...")
    data = fetch_live_data(tickers)

    # Create a list to store the price and action information
    price_data = []

    for ticker in tickers:
        if ticker in data and not data[ticker].empty:
            df = data[ticker]

            # Get the most recent closing price
            current_price = df['Close'].iloc[-1]

            # Perform feature enhancement
            df = enhance_features(df)

            if len(df) < 60:
                st.warning(
                    f"Not enough data after feature enhancement for {ticker}. Rows: {len(df)}")
                st.info(
                    "Consider selecting a different time frame or checking data availability.")
                continue

            # Reset index without dropping to keep 'date' for plotting
            df.reset_index(inplace=True)

            # Add Transformer predictions (currently placeholder)
            df = add_transformer_predictions(df)

            # Prepare data for RL
            data_scaled, scaler = prepare_rl_data(df)

            # Initialize the custom trading environment with news sentiment and economic indicator
            env = TradingEnv(data_scaled, sentiment_score, economic_indicator,
                             take_profit=take_profit, stop_loss=stop_loss)

            # Initialize a list to hold the predictions from different models
            model_predictions = []

            for rl_algorithm, algo_cls in [('A2C', A2C), ('PPO', PPO), ('DDPG', DDPG)]:
                try:
                    model_path = os.path.join(
                        "models", f"{ticker}_{rl_algorithm}_rl_trading_agent.zip")

                    # ---------- NEW ----------
                    model = safe_load(model_path, algo_cls)
                    if model is None:          # skip tickers whose model couldn’t be opened
                        continue
                    # ---------- END NEW ------

                    # Get the latest data point
                    latest_data = data_scaled.iloc[-1]

                    # Predict
                    action_value = get_action_value(
                        model, latest_data, sentiment_score, economic_indicator)

                    decision, confidence = map_action_to_decision_confidence(action_value)

                    model_predictions.append({
                        'Model': rl_algorithm,
                        'Action Value': action_value,
                        'Decision': decision,
                        'Confidence': confidence
                    })

                except (ValueError, FileNotFoundError, OSError) as e:
                    st.warning(f"Model file {model_path} could not be loaded: {e}")
                    continue

            if not model_predictions:
                # If no predictions were made
                final_decision = "N/A"
                final_confidence = "N/A"
                action_values = {'A2C': 'N/A', 'PPO': 'N/A', 'DDPG': 'N/A'}
                decisions_dict = {'A2C': 'N/A',
                                  'PPO': 'N/A', 'DDPG': 'N/A'}
                confidences_dict = {'A2C': 'N/A',
                                    'PPO': 'N/A', 'DDPG': 'N/A'}
            else:
                # Prepare dictionaries to hold model-specific predictions
                action_values = {}
                decisions_dict = {}
                confidences_dict = {}
                for pred in model_predictions:
                    action_values[pred['Model']] = f"{pred['Action Value']:.2f}"
                    decisions_dict[pred['Model']] = pred['Decision']
                    confidences_dict[pred['Model']] = pred['Confidence']

                # Determine final recommendation based on voting
                from collections import Counter
                decisions = [pred['Decision'] for pred in model_predictions]
                decision_counts = Counter(decisions)
                final_decision = decision_counts.most_common(1)[0][0]

                # Calculate average confidence for final decision
                confidence_levels = {'High': 3,
                                     'Medium': 2, 'Low': 1, 'Neutral': 0}
                confidences_for_final_decision = [
                    pred['Confidence'] for pred in model_predictions if pred['Decision'] == final_decision]
                confidence_values = [
                    confidence_levels.get(conf, 0) for conf in confidences_for_final_decision]
                avg_confidence_value = sum(confidence_values) / \
                    len(confidence_values) if confidence_values else 0
                if avg_confidence_value >= 2.5:
                    final_confidence = 'High'
                elif avg_confidence_value >= 1.5:
                    final_confidence = 'Medium'
                elif avg_confidence_value >= 0.5:
                    final_confidence = 'Low'
                else:
                    final_confidence = 'Neutral'

            # Adjust take profit and stop loss prices based on the final_decision
            if final_decision == "Buy":
                # For long positions
                take_profit_price = current_price * (1 + take_profit)
                stop_loss_price = current_price * (1 - stop_loss)
            elif final_decision == "Sell":
                # For short positions
                take_profit_price = current_price * (1 - take_profit)
                stop_loss_price = current_price * (1 + stop_loss)
            else:
                # For Hold or N/A, use N/A for prices
                take_profit_price = "N/A"
                stop_loss_price = "N/A"

            # Format the take profit and stop loss prices if they are numbers
            if isinstance(take_profit_price, float):
                take_profit_price_formatted = f"${take_profit_price:.2f}"
            else:
                take_profit_price_formatted = take_profit_price

            if isinstance(stop_loss_price, float):
                stop_loss_price_formatted = f"${stop_loss_price:.2f}"
            else:
                stop_loss_price_formatted = stop_loss_price

            # Create a record for this ticker including the model predictions
            ticker_data = {
                'Ticker': ticker,
                'Current Price': f"${current_price:.2f}",
                'Take Profit Price': take_profit_price_formatted,
                'Stop Loss Price': stop_loss_price_formatted,
                'Final Recommended Action': final_decision,
                'Final Confidence': final_confidence
            }

            # Add model predictions to the record
            for model_name in ['A2C', 'PPO', 'DDPG']:
                ticker_data[f'{model_name} Action Value'] = action_values.get(
                    model_name, 'N/A')
                ticker_data[f'{model_name} Decision'] = decisions_dict.get(
                    model_name, 'N/A')
                ticker_data[f'{model_name} Confidence'] = confidences_dict.get(
                    model_name, 'N/A')

            # Append the data to the list
            price_data.append(ticker_data)

        else:
            st.warning(f"No data for {ticker}, skipping.")

    # Create a DataFrame from the list
    price_df = pd.DataFrame(price_data)

    # Check if the DataFrame is not empty
    if not price_df.empty:
        # Split price_df into two tables
        
        # First table with price information and final recommendations
        df_prices = price_df[['Ticker', 'Current Price', 'Take Profit Price', 'Stop Loss Price',
                              'Final Recommended Action', 'Final Confidence']]
        
        # Second table with RL model predictions
        df_model_predictions = price_df[['Ticker',
                                         'A2C Action Value', 'A2C Decision', 'A2C Confidence',
                                         'PPO Action Value', 'PPO Decision', 'PPO Confidence',
                                         'DDPG Action Value', 'DDPG Decision', 'DDPG Confidence']]
        
        # Display the first table
        st.subheader("Price Information and Final Recommendations")
        st.dataframe(df_prices)
        
        # Now split df_model_predictions into two tables
        
        # First RL Predictions Table: Actions and Decisions
        df_model_actions = df_model_predictions[['Ticker',
                                                 'A2C Action Value', 'A2C Decision',
                                                 'PPO Action Value', 'PPO Decision',
                                                 'DDPG Action Value', 'DDPG Decision']]
        
        # Second RL Predictions Table: Confidence Levels
        df_model_confidence = df_model_predictions[['Ticker',
                                                    'A2C Confidence',
                                                    'PPO Confidence',
                                                    'DDPG Confidence']]
        
        # Display the RL Model Actions and Decisions Table
        st.subheader("RL Model Actions and Decisions")
        st.dataframe(df_model_actions)
        
        # Display the RL Model Confidence Levels Table
        st.subheader("RL Model Confidence Levels")
        st.dataframe(df_model_confidence)
    else:
        st.info("No price data available to display.")
        
    return df_prices, df_model_actions, df_model_confidence

if __name__ == "__main__":
    main()