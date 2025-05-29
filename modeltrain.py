import os
import joblib
import requests
import pandas as pd
import numpy as np

# Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, TimeDistributed, Flatten, LSTM, GRU, concatenate
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Optional plotting
import matplotlib.pyplot as plt

# Reinforcement Learning Libraries
import gym
from gym import spaces
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

# Sentiment Analysis
from transformers import pipeline

# =============================================================================
# 1) Data Fetching
# =============================================================================
def fetch_live_data(tickers, retries=3):
    """
    Fetch 15-minute historical data from FinancialModelingPrep.
    Requires FMP_API_KEY to be set in your environment.
    """
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set FMP_API_KEY in your environment.")

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
                    raise e

    return data

# =============================================================================
# 2) Fetch News Data
# =============================================================================
def fetch_news_data():
    """
    Fetch news data from NewsAPI and economic data from FRED API.
    Requires NEWS_API_KEY_NEWSAPI and NEWS_API_KEY_FRED to be set in your environment.
    """
    news_api_key = os.getenv("NEWS_API_KEY_NEWSAPI")
    fred_api_key = os.getenv("NEWS_API_KEY_FRED")

    if not news_api_key:
        raise ValueError("NewsAPI key not found. Please set NEWS_API_KEY_NEWSAPI in your environment.")
    if not fred_api_key:
        raise ValueError("FRED API key not found. Please set NEWS_API_KEY_FRED in your environment.")

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

    # Extract recent economic indicator
    fred_observations = fred_data.get('observations', [])
    latest_fred_value = float(fred_observations[-1]['value']) if fred_observations else 0

    return headlines, latest_fred_value

# =============================================================================
# 3) Sentiment Analysis of News Headlines
# =============================================================================
def compute_sentiment_score(headlines):
    """
    Compute average sentiment polarity of the news headlines using BERT transformer.
    """
    if not headlines:
        return 0  # Neutral sentiment if no headlines

    # Initialize the sentiment analyzer
    sentiment_analyzer = pipeline('sentiment-analysis')  # Using BERT transformer

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
            print(f"Error analyzing sentiment for headline: {headline}. Error: {e}")
            sentiment_scores.append(0)

    average_sentiment = np.mean(sentiment_scores)
    return average_sentiment

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
    df['BB_Bandwidth'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
    return df

def compute_OBV(df):
    """Compute On-Balance Volume."""
    df['Delta_Close'] = df['Close'].diff()
    df['Direction'] = np.where(df['Delta_Close'] > 0, 1, np.where(df['Delta_Close'] < 0, -1, 0))
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
    df['TR'] = df[['High', 'Close']].shift(1).max(axis=1) - df[['Low', 'Close']].shift(1).min(axis=1)

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
    df['DX'] = (abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14'])) * 100
    df['ADX'] = df['DX'].rolling(window=period).mean()

    df.drop(columns=['TR', '+DM', '-DM', 'TR14', '+DM14', '-DM14', '+DI14', '-DI14', 'DX'], inplace=True)
    df.dropna(inplace=True)
    return df

# =============================================================================
# 5) Enhanced Feature Engineering
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
# 6) Prepare Data for the Transformer Model
# =============================================================================
def prepare_transformer_data(df, input_sequence_length=60, forecast_horizon=1):
    """
    Prepare data in sequences for training the Transformer model.
    """
    data = df.copy()
    # We'll use the 'Close' price as target and other features as inputs.
    feature_columns = [col for col in data.columns if col != 'Close' and col != 'date']  # Exclude target and date
    target_column = 'Close'
    X = []
    y = []
    for i in range(len(data) - input_sequence_length - forecast_horizon +1):
        # Input sequence
        X_seq = data[feature_columns].iloc[i:(i + input_sequence_length)].values
        # Target: next close price after the input sequence
        y_seq = data[target_column].iloc[(i + input_sequence_length):(i + input_sequence_length + forecast_horizon)].values
        X.append(X_seq)
        y.append(y_seq)
    X = np.array(X)
    y = np.array(y)
    return X, y

# =============================================================================
# 7) Implement and Train the Transformer Model
# =============================================================================
def build_transformer_model(input_shape, d_model=64, num_heads=4, num_layers=2, dropout_rate=0.1):
    """
    Build a Transformer model for time series forecasting.
    """
    inputs = Input(shape=input_shape)

    # Project inputs to d_model dimensions using a Dense layer
    x = Dense(d_model)(inputs)

    for _ in range(num_layers):
        # Self-attention layer
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        # Feed-forward network
        ffn_output = Dense(d_model, activation='relu')(x)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Flatten and output layer
    x = Flatten()(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_ensemble_model(input_shape):
    """
    Build an ensemble model combining LSTM and GRU.
    """
    inputs = Input(shape=input_shape)
    
    # LSTM Branch
    lstm_out = LSTM(64, return_sequences=False)(inputs)
    
    # GRU Branch
    gru_out = GRU(64, return_sequences=False)(inputs)
    
    # Concatenate outputs
    combined = concatenate([lstm_out, gru_out])
    
    # Dense layers
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_transformer_model(X_train, y_train, X_val, y_val, input_shape):
    """
    Build and train the Transformer model.
    """
    # Build the model
    model = build_transformer_model(input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    return model

def train_ensemble_model(X_train, y_train, X_val, y_val, input_shape):
    """
    Build and train the ensemble model.
    """
    # Build the model
    model = build_ensemble_model(input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    return model

# =============================================================================
# 8) Use Transformer's Predictions as Inputs to RL
# =============================================================================
def add_transformer_predictions(df, model, scaler, input_sequence_length=60):
    """
    Use the Transformer model to predict future prices and add predictions as a new column.
    """
    data = df.copy()
    feature_columns = [col for col in data.columns if col != 'Close' and col != 'date']  # Exclude target and date
    predictions = []
    for i in range(len(data) - input_sequence_length):
        X_seq = data[feature_columns].iloc[i:i + input_sequence_length].values
        X_seq = np.expand_dims(X_seq, axis=0)
        pred = model.predict(X_seq)
        # Inverse transform the prediction
        # Since we scaled the data, we need to inverse transform the predictions
        inv_scaled = np.zeros((1, len(feature_columns)+1))
        inv_scaled[0, -1] = pred[0][0]
        pred_inverse = scaler.inverse_transform(inv_scaled)[0][-1]
        predictions.append(pred_inverse)
    # Pad the beginning of the predictions to align with the data
    predictions = [np.nan]*input_sequence_length + predictions
    data['Transformer_Prediction'] = predictions
    # Drop NaN values (if any)
    data.dropna(inplace=True)
    return data

# =============================================================================
# 9) Custom Trading Environment for RL (Updated with Transformer's predictions)
# =============================================================================
class TradingEnv(gym.Env):
    """A custom trading environment for RL."""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, sentiment_score, economic_indicator, take_profit=0.05, stop_loss=0.05):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.total_steps = len(df) - 1
        # Actions: Continuous action between -1 and 1 (sell, hold, buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Observations: Price data, indicators, sentiment, economic data, Transformer's prediction
        num_features = len(self.df.columns) - 1  # Exclude 'date' column
        num_external_factors = 3  # sentiment_score, economic_indicator, Transformer's prediction
        total_obs_size = num_features + num_external_factors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )
        # Initial balance and holdings
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.shares_held = 0
        self.last_price = self.df.loc[self.current_step, 'Close']
        # External factors
        self.sentiment_score = sentiment_score
        self.economic_indicator = economic_indicator
        # Risk management parameters
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        # Trade logging
        self.trade_history = []

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.shares_held = 0
        self.trade_history = []
        self.last_price = self.df.loc[self.current_step, 'Close']
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step].drop(['date']).values.astype('float32')
        # Append external factors
        obs = np.append(obs, [self.sentiment_score, self.economic_indicator, self.df.loc[self.current_step, 'Transformer_Prediction']])
        # Check for NaN or infinite values in observations
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"Warning: Observation contains invalid values at step {self.current_step}. Replacing with zeros.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        reward = 0
        done = False

        # Ensure current_price is valid
        if np.isnan(current_price) or current_price <= 0:
            print(f"Warning: Current price is invalid at step {self.current_step}. Adjusting to a small positive value.")
            current_price = 1e-6  # Prevent division by zero

        # Ensure action is scalar
        if isinstance(action, np.ndarray):
            action = action[0]

        # Clip action to the valid range
        action = np.clip(action, -1, 1)

        # Implement position sizing based on action value
        if action > 0:
            # Buy
            amount_to_invest = self.balance * action
            shares_bought = amount_to_invest / current_price

            # Check for division by zero or invalid shares_bought
            if np.isnan(shares_bought) or np.isinf(shares_bought):
                print(f"Warning: Invalid shares_bought at step {self.current_step}. Skipping buy action.")
                shares_bought = 0.0

            if shares_bought > 0:
                self.balance -= shares_bought * current_price
                self.shares_held += shares_bought
        elif action < 0:
            # Sell
            fraction_to_sell = -action
            shares_to_sell = self.shares_held * fraction_to_sell

            # Check for invalid shares_to_sell
            if np.isnan(shares_to_sell) or np.isinf(shares_to_sell):
                print(f"Warning: Invalid shares_to_sell at step {self.current_step}. Skipping sell action.")
                shares_to_sell = 0.0

            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
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
                self.shares_held = 0
            done = True  # End episode

        # Reward: change in net worth adjusted for risk
        reward = self.net_worth - prev_net_worth

        # Prevent NaN rewards
        if np.isnan(reward) or np.isinf(reward):
            print(f"Warning: Invalid reward at step {self.current_step}. Setting reward to 0.")
            reward = 0.0

        self.last_price = current_price
        self.current_step += 1

        if self.current_step >= self.total_steps:
            done = True

        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        # Print the current balance and net worth
        profit = self.net_worth - self.initial_balance
        print(f"Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}")

# =============================================================================
# 10) Prepare Data for RL
# =============================================================================
def prepare_rl_data(df):
    """Scale and prepare data for the RL environment."""
    data = df.copy()
    # Adjust the scaler to avoid issues with zero values
    scaler = MinMaxScaler(feature_range=(1e-6, 1))  # Avoid zero in scaling
    # Exclude 'date' from features
    features = data.drop(columns=['date'])
    scaled_features = scaler.fit_transform(features)
    # Create DataFrame with scaled features
    data_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    # Add 'date' column back
    data_scaled['date'] = data['date'].values
    # Reorder columns
    cols = ['date'] + features.columns.tolist()
    data_scaled = data_scaled[cols]
    return data_scaled, scaler

# =============================================================================
# 11) Training the RL Agent
# =============================================================================
def train_rl_agent(env, algorithm='A2C'):
    """Train an RL agent using specified algorithm."""
    # Vectorized environment
    env_vec = DummyVecEnv([lambda: env])
    # Initialize the agent with tuned hyperparameters
    if algorithm == 'A2C':
        model = A2C('MlpPolicy', env_vec, verbose=1, learning_rate=0.0007, n_steps=5, gamma=0.99, gae_lambda=0.95)
    elif algorithm == 'PPO':
        model = PPO('MlpPolicy', env_vec, verbose=1, learning_rate=0.0003, n_steps=2048, gamma=0.99, gae_lambda=0.95)
    elif algorithm == 'DDPG':
        model = DDPG('MlpPolicy', env_vec, verbose=1, learning_rate=0.001, gamma=0.99)
    else:
        raise ValueError("Unsupported algorithm")
    # Train the agent
    model.learn(total_timesteps=10000)
    return model

# =============================================================================
# 12) Performance Monitoring
# =============================================================================
def evaluate_model(env, model):
    """Evaluate the RL agent's performance."""
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print(f"Total Profit: {env.net_worth - env.initial_balance:.2f}")

# =============================================================================
# 13) Automate Execution (Integration Placeholder)
# =============================================================================
def execute_trades(model, latest_data, sentiment_score, economic_indicator):
    """
    Placeholder function for automating trade execution.
    In a live trading system, this function would interface with a brokerage API.
    """
    # Prepare latest data point
    obs = latest_data.drop(['date']).values.astype('float32')
    obs = np.append(obs, [sentiment_score, economic_indicator, latest_data['Transformer_Prediction']])  # Include external factors
    # Check for NaN or infinite values in observations
    if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
        print("Warning: Observation contains invalid values for trade execution.")
        obs = np.nan_to_num(obs)
    # Predict action
    action, _states = model.predict(obs)
    # Map action to trade execution
    if action > 0.01:
        print(f"Recommended Action: Buy {action[0]*100:.2f}% of available balance")
        # Place buy order via brokerage API
    elif action < -0.01:
        print(f"Recommended Action: Sell {-action[0]*100:.2f}% of holdings")
        # Place sell order via brokerage API
    else:
        print("Recommended Action: Hold")
        # Hold - do nothing

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    tickers = ["CL=F", "^VIX", "ACB", "CGC", "CL=F", "TLRY"]  # Example tickers
    # Fetch external data
    headlines, economic_indicator = fetch_news_data()
    sentiment_score = compute_sentiment_score(headlines)
    # 1) Fetch data
    data = fetch_live_data(tickers)
    for ticker in tickers:
        if ticker not in data or data[ticker].empty:
            print(f"No data for {ticker}, skipping.")
            continue
        df = data[ticker]
        print(f"\nFetched {len(df)} rows of data for {ticker}.")
        # Remove rows where 'Close' price is zero or negative
        df = df[df['Close'] > 0]
        df.reset_index(inplace=True)
        if df.empty:
            print(f"No valid data for {ticker} after removing zero prices, skipping.")
            continue
        # 2) Enhance features with advanced indicators
        df = enhance_features(df)
        print(f"After enhance_features, {len(df)} rows remain (NaNs dropped).")
        # Check if we have enough data
        if len(df) < 100:
            print(f"Not enough data after feature enhancement. Rows: {len(df)}")
            continue
        # 3) Prepare data for Transformer
        # Scale data first
        scaler = MinMaxScaler()
        features = df.drop(columns=['date'])
        scaled_features = scaler.fit_transform(features)
        df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
        df_scaled['date'] = df['date'].values
        cols = ['date'] + features.columns.tolist()
        df_scaled = df_scaled[cols]
        # Prepare sequences for Transformer
        X, y = prepare_transformer_data(df_scaled)
        # Split data into training and validation sets
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        input_shape = X_train.shape[1:]  # (sequence_length, num_features)
        # 4) Train the Transformer model
        print("\n[ Training the Transformer Model ]")
        transformer_model = train_transformer_model(X_train, y_train, X_val, y_val, input_shape)
        # 5) Train the Ensemble model
        print("\n[ Training the Ensemble Model ]")
        ensemble_model = train_ensemble_model(X_train, y_train, X_val, y_val, input_shape)
        # 6) Use Transformer's predictions as inputs to RL
        df_with_preds = add_transformer_predictions(df_scaled, transformer_model, scaler)
        # 7) Prepare data for RL
        data_for_rl = df_with_preds.copy()
        # Note: The scaling was already done, so no need to rescale
        # 8) Initialize the custom trading environment with the Transformer's predictions, news sentiment, and economic indicator
        env = TradingEnv(data_for_rl, sentiment_score, economic_indicator)
        # 9) Train the RL agent with different algorithms
        for algorithm in ['A2C', 'PPO', 'DDPG']:
            print(f"\n[ Training the RL Agent using {algorithm} ]")
            model = train_rl_agent(env, algorithm=algorithm)
            # 10) Save the trained model
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"{ticker}_{algorithm}_rl_trading_agent.zip")
            model.save(model_path)
            print(f"{algorithm} trading agent saved to {model_path}")
            # 11) Evaluate the model
            print(f"\n[ Evaluating the RL Agent using {algorithm} ]")
            evaluate_model(env, model)
            # 12) Backtesting across multiple datasets (simulated by splitting data)
            print(f"\n[ Backtesting the {algorithm} Agent ]")
            num_periods = 3
            period_length = len(data_for_rl) // num_periods
            for i in range(num_periods):
                start_idx = i * period_length
                end_idx = (i + 1) * period_length if i < num_periods - 1 else len(data_for_rl)
                backtest_data = data_for_rl.iloc[start_idx:end_idx]
                backtest_env = TradingEnv(backtest_data, sentiment_score, economic_indicator)
                evaluate_model(backtest_env, model)
            # 13) Automate execution (Placeholder)
            print("\n[ Automating Trade Execution ]")
            latest_data = data_for_rl.iloc[-1]
            execute_trades(model, latest_data, sentiment_score, economic_indicator)
            print(f"\nProcessing completed for ticker: {ticker} using {algorithm}")