import streamlit as st
import yfinance as yf
import numpy as np
from datetime import datetime

# --- Helper Functions with Caching ---

@st.cache_data(ttl=1800)  # Cache data for 30 minutes
def get_stock_data(ticker_symbol):
    """
    Fetches stock data from yfinance, calculates annualized volatility,
    and returns key serializable information.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # A simple check to see if the ticker is valid
        if 'longName' not in info or info['longName'] is None:
            return None, None, None, None

        # Get 1 year of historical data for volatility calculation
        hist = ticker.history(period="1y")
        if hist.empty:
            return None, None, None, None
            
        S0 = hist['Close'].iloc[-1]
        long_name = info['longName']

        # Calculate Annualized Volatility
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        # Use 252 trading days for annualization
        sigma = log_returns.std() * np.sqrt(252)

        expirations = ticker.options

        # Return only serializable data
        return S0, sigma, expirations, long_name
    except Exception:
        return None, None, None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_risk_free_rate():
    """
    Fetches the 13-week US Treasury Bill yield (^IRX) as a proxy
    for the risk-free rate.
    """
    try:
        t_bill = yf.Ticker("^IRX")
        rate = t_bill.history(period="1d")['Close'].iloc[-1]
        return rate / 100
    except Exception:
        # Provide a fallback rate if the API fails
        st.warning("Could not fetch live risk-free rate. Defaulting to 5%.")
        return 0.05

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(layout="wide")
st.title("Binomial Options Calculator (with Yahoo Finance Data)")
st.markdown("Select a stock, and the app will fetch live data to price the option using your custom formulas.")

# --- Stock Selection Input ---
# MODIFIED: Added specific instructions for international stocks
st.subheader("1. Select a Stock")
ticker_symbol = st.text_input(
    "Enter Stock Ticker", 
    "AAPL",
    help="For non-US stocks, add the exchange suffix. Examples: RELIANCE.NS (India), VOW3.DE (Germany), BHP.AX (Australia)."
).upper()

if ticker_symbol:
    # --- Data Fetching and Display ---
    S0, sigma, expirations, long_name = get_stock_data(ticker_symbol)

    if S0 is None:
        st.error(f"Invalid or unsupported ticker symbol: {ticker_symbol}. Please check the symbol and add a market suffix if needed (e.g., .NS for India).")
    else:
        st.header(f"Selected: {long_name} ({ticker_symbol})")
        r = get_risk_free_rate()

        # Display key fetched data points
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Stock Price (S₀)", f"${S0:.2f}")
        col2.metric("Annualized Volatility (σ)", f"{sigma:.2%}")
        col3.metric("Risk-Free Rate (r)", f"{r:.2%}")

        st.divider()

        # --- Option Parameter Selection ---
        st.subheader("2. Select Option Parameters")

        # MODIFIED: Added a more helpful message when no options are found.
        if not expirations:
            st.warning(
                f"This stock ({ticker_symbol}) has no available option expiration dates on Yahoo Finance. "
                "This is common for certain stocks or if the ticker symbol is for a non-US market without the correct suffix (e.g., '.NS' for India)."
            )
        else:
            # Create the Ticker object again (this is not cached and is lightweight)
            # to get the option chain data.
            ticker_obj = yf.Ticker(ticker_symbol)
            
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            with sub_col1:
                option_type = st.selectbox("Option Type", ('Call', 'Put'))
            with sub_col2:
                selected_expiry = st.selectbox("Expiration Date", expirations)
            
            # Calculate Time to Expiration in years
            expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
            T = (expiry_date - datetime.now()).days / 365.0

            # Fetch the option chain for the selected expiry date
            option_chain = ticker_obj.option_chain(selected_expiry)
            
            if option_type == 'Call':
                strikes = option_chain.calls['strike'].tolist()
            else:
                strikes = option_chain.puts['strike'].tolist()
            
            # Automatically select a strike price near the current stock price as a default
            closest_strike = min(strikes, key=lambda x: abs(x - S0))
            default_strike_index = strikes.index(closest_strike)

            with sub_col3:
                K = st.selectbox("Strike Price (K)", strikes, index=default_strike_index)
            
            st.info(f"Time to Expiration (T) = **{T:.3f} years** ({int(T * 365)} days)", icon="⏳")

            # --- CALCULATION LOGIC BASED ON YOUR INSTRUCTIONS ---
            st.divider()
            st.header("3. Calculation Results")

            # 1. Calculation of Up and Down factor
            u = 1 + sigma
            d = 1 / u
            
            price_up = S0 * u
            price_down = S0 * d

            # 2. Risk Neutral Probability
            try:
                # The growth factor for the period is (1 + r * T)
                prob_up = ((1 + r * T) - d) / (u - d)
                prob_down = 1 - prob_up
                
                # Check for arbitrage opportunities
                if not (0 <= prob_up <= 1):
                    st.warning(
                        f"Arbitrage Opportunity Detected! "
                        f"The calculated probability ({prob_up:.2f}) is outside the valid [0, 1] range. "
                        f"Results may be unreliable."
                    )
                
                # 3. Option valuation at final nodes
                if option_type == 'Call':
                    payoff_up = max(0, price_up - K)
                    payoff_down = max(0, price_down - K)
                else:  # Put Option
                    payoff_up = max(0, K - price_up)
                    payoff_down = max(0, K - price_down)

                # Combine valuation and discounting per your formula
                total_value_final_node = (payoff_up * prob_up) + (payoff_down * prob_down)

                # 4. Discounting future values
                option_price = total_value_final_node / (1 + r * T)

                # --- Display the results ---
                st.metric(label=f"Calculated {option_type} Option Price", value=f"${option_price:.4f}")

                st.subheader("Intermediate Values (Your Formulas)")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.write(f"**Up Factor (u):** `{u:.4f}`")
                    st.write(f"**Down Factor (d):** `{d:.4f}`")
                    st.write(f"**Stock Price (Up):** `${S0 * u:.2f}`") # Using S0*u directly for display
                    st.write(f"**Stock Price (Down):** `${S0 * d:.2f}`") # Using S0*d directly for display

                with res_col2:
                    st.write(f"**Probability of Up Move (p):** `{prob_up:.4f}`")
                    st.write(f"**Probability of Down Move (1-p):** `{prob_down:.4f}`")
                    st.write(f"**Payoff (Up):** `${payoff_up:.2f}`")
                    st.write(f"**Payoff (Down):** `${payoff_down:.2f}`")

            except ZeroDivisionError:
                st.error("Calculation error: Up and Down factors are equal (u=d). Cannot calculate probability.")

