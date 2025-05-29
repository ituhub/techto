import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from enhance import main as run_enhance_app  # Import your real logic

# Page configuration
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

# Apply custom CSS...
st.markdown("""
    <style>
        .dataframe-container {
            zoom: 0.85 !important;
        }
        .stDataFrame > div > div > div > div {
            font-size: 14px !important;
        }

        /* Sidebar and heading text to Navy Blue */
        .css-1d391kg, .css-qbe2hs, .css-10trblm, .css-1v0mbdj, .css-1v3fvcr,
        .css-1h0z5md h1, .css-1h0z5md h2, .css-1h0z5md h3, .css-1h0z5md h4, .css-1h0z5md h5, .css-1h0z5md h6,
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            color: #000080 !important;
        }

        /* ✅ Selected tickers in multiselect input */
        div[data-baseweb="tag"] {
            background-color: #000080 !important;
            color: white !important;
            border-radius: 5px;
            font-weight: 500;
            padding: 4px 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "About"],
        icons=["graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

# Dashboard Page
if selected == "Dashboard":
    st.title("📊 AI-Powered Trading Forecast")

    # Run the actual enhance logic and retrieve the DataFrames
    df_prices, df_model_actions, df_model_confidence = run_enhance_app()

    st.markdown("### 📈 Price Forecast & Recommendations")
    # Color only the font of the Final Recommended Action column
    def color_action_font(val):
        if val == 'Buy':
            return 'color: green'
        elif val == 'Sell':
            return 'color: red'
        elif val == 'Hold':
            return 'color: orange'
        return ''

    styled_prices = df_prices.style.applymap(color_action_font, subset=['Final Recommended Action'])
    st.dataframe(styled_prices, use_container_width=True)

    # Optional: Add a simple candlestick chart if the data exists
    if 'Date' in df_prices.columns and 'Current Price' in df_prices.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_prices['Date'], y=df_prices['Current Price'], name='Current Price'))
        fig.update_layout(
            template="plotly_dark",
            title="📊 Price Trend",
            xaxis_title="Date",
            yaxis_title="Price",
            font=dict(color="#000080"),
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🤖 Model Action Decisions")
    with st.expander("View Model Action Table"):
        st.dataframe(df_model_actions, use_container_width=True)

    st.markdown("### 📊 Model Confidence Levels")
    with st.expander("View Model Confidence Table"):
        st.dataframe(df_model_confidence, use_container_width=True)

    # Export Option
    st.download_button("Download Forecast Table as CSV", df_prices.to_csv(index=False), file_name="forecast_table.csv")

# About Page
elif selected == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
        This AI-powered trading dashboard provides actionable market insights using reinforcement learning models (A2C, PPO, DDPG).

        **Features**:
        - Real-time predictions via FMP API
        - Price forecasting and model decision aggregation
        - Confidence evaluation and trade suggestions

        Built with ❤️ using Streamlit.
    """)
