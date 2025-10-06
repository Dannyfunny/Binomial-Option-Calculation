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

# --- CORE CALCULATION FUNCTIONS ---

def calculate_option_price_custom(S0, K, T, r, sigma, option_type):
    """Calculates the option price based on the user's custom single-step binomial model."""
    if T <= 0 or sigma <= 0: # Avoid invalid calculations
        return 0, 0, 0, 0, 0, 0, 0, 0

    # 1. Up and Down factors
    u = 1 + sigma
    d = 1 / u
    
    # Check for division by zero
    if u == d:
        return 0, 0, 0, 0, 0, 0, 0, 0
    
    price_up = S0 * u
    price_down = S0 * d

    # 2. Risk Neutral Probability
    prob_up = ((1 + r * T) - d) / (u - d)
    prob_down = 1 - prob_up

    # 3. Option valuation at final nodes
    if option_type == 'Call':
        payoff_up = max(0, price_up - K)
        payoff_down = max(0, price_down - K)
    else:  # Put Option
        payoff_up = max(0, K - price_up)
        payoff_down = max(0, K - price_down)

    # 4. Discounting
    total_value_final_node = (payoff_up * prob_up) + (payoff_down * prob_down)
    option_price = total_value_final_node / (1 + r * T)
    
    return option_price, u, d, prob_up, prob_down, price_up, price_down, payoff_up, payoff_down

def calculate_greeks(S0, K, T, r, sigma, option_type):
    """
    Calculates option Greeks using the finite difference method.
    """
    # Small changes for derivatives
    dS = S0 * 0.01   # 1% change in stock price
    dSigma = 0.01    # 1% change in volatility
    dT = 1 / 365.0   # 1 day change in time
    dR = 0.01        # 1% change in risk-free rate

    # Base price
    base_price, *_ = calculate_option_price_custom(S0, K, T, r, sigma, option_type)

    # Prices for Delta and Gamma calculation
    price_plus_S, *_ = calculate_option_price_custom(S0 + dS, K, T, r, sigma, option_type)
    price_minus_S, *_ = calculate_option_price_custom(S0 - dS, K, T, r, sigma, option_type)
    
    # Price for Vega calculation
    price_plus_sigma, *_ = calculate_option_price_custom(S0, K, T, r, sigma + dSigma, option_type)
    
    # Price for Theta calculation
    price_minus_T, *_ = calculate_option_price_custom(S0, K, T - dT, r, sigma, option_type)
    
    # Price for Rho calculation
    price_plus_r, *_ = calculate_option_price_custom(S0, K, T, r + dR, sigma, option_type)

    # Greek Calculations
    delta = (price_plus_S - price_minus_S) / (2 * dS)
    gamma = (price_plus_S - 2 * base_price + price_minus_S) / (dS ** 2)
    vega = (price_plus_sigma - base_price) / (dSigma * 100) # Per 1% change
    theta = (price_minus_T - base_price) / dT # Per day
    rho = (price_plus_r - base_price) / (dR * 100) # Per 1% change
    
    return delta, gamma, vega, theta, rho

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(layout="wide")
st.title("Binomial Options Calculator (with Yahoo Finance Data)")
st.markdown("Select a stock, and the app will fetch live data to price the option using your custom formulas.")

# --- Stock Selection Input ---
st.subheader("1. Select a Stock")
ticker_symbol = st.text_input(
    "Enter Stock Ticker", 
    "AAPL",
    help="For non-US stocks, add the exchange suffix. Examples: RELIANCE.NS (India), VOW3.DE (Germany), BHP.AX (Australia)."
).upper()

if ticker_symbol:
    S0, sigma, expirations, long_name = get_stock_data(ticker_symbol)

    if S0 is None:
        st.error(f"Invalid or unsupported ticker symbol: {ticker_symbol}. Please check the symbol and add a market suffix if needed (e.g., .NS for India).")
    else:
        st.header(f"Selected: {long_name} ({ticker_symbol})")
        r = get_risk_free_rate()

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Stock Price (S₀)", f"${S0:.2f}")
        col2.metric("Annualized Volatility (σ)", f"{sigma:.2%}")
        col3.metric("Risk-Free Rate (r)", f"{r:.2%}")
        st.divider()

        st.subheader("2. Select Option Parameters")
        if not expirations:
            st.warning(f"This stock ({ticker_symbol}) has no available option expiration dates on Yahoo Finance. This is common for certain stocks or if the ticker symbol is for a non-US market without the correct suffix (e.g., '.NS' for India).")
        else:
            ticker_obj = yf.Ticker(ticker_symbol)
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            with sub_col1:
                option_type = st.selectbox("Option Type", ('Call', 'Put'))
            with sub_col2:
                selected_expiry = st.selectbox("Expiration Date", expirations)
            
            expiry_date = datetime.strptime(selected_expiry, '%Y-%m-%d')
            T = (expiry_date - datetime.now()).days / 365.0

            option_chain = ticker_obj.option_chain(selected_expiry)
            strikes = option_chain.calls['strike'].tolist() if option_type == 'Call' else option_chain.puts['strike'].tolist()
            
            closest_strike = min(strikes, key=lambda x: abs(x - S0))
            default_strike_index = strikes.index(closest_strike)

            with sub_col3:
                K = st.selectbox("Strike Price (K)", strikes, index=default_strike_index)
            
            st.info(f"Time to Expiration (T) = **{T:.3f} years** ({int(T * 365)} days)", icon="⏳")
            st.divider()
            
            st.header("3. Calculation Results")
            # --- Perform Main Calculation ---
            (option_price, u, d, prob_up, prob_down, 
             price_up, price_down, payoff_up, payoff_down) = calculate_option_price_custom(S0, K, T, r, sigma, option_type)

            st.metric(label=f"Calculated {option_type} Option Price", value=f"${option_price:.4f}")

            # --- Perform Greeks Calculation ---
            delta, gamma, vega, theta, rho = calculate_greeks(S0, K, T, r, sigma, option_type)
            st.subheader("Option Greeks (Estimates)")
            
            greek_col1, greek_col2, greek_col3, greek_col4, greek_col5 = st.columns(5)
            greek_col1.metric("Delta", f"{delta:.4f}")
            greek_col2.metric("Gamma", f"{gamma:.4f}")
            greek_col3.metric("Vega", f"{vega:.4f}")
            greek_col4.metric("Theta", f"{theta:.4f}")
            greek_col5.metric("Rho", f"{rho:.4f}")

            st.divider()

            # --- Display Intermediate Values ---
            st.subheader("Intermediate Values (Your Formulas)")
            if not (0 <= prob_up <= 1):
                st.warning(f"Arbitrage Opportunity Detected! The calculated probability ({prob_up:.2f}) is outside the valid [0, 1] range. Results may be unreliable.")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.write(f"**Up Factor (u):** `{u:.4f}`")
                st.write(f"**Down Factor (d):** `{d:.4f}`")
                st.write(f"**Stock Price (Up):** `${price_up:.2f}`")
                st.write(f"**Stock Price (Down):** `${price_down:.2f}`")
            with res_col2:
                st.write(f"**Probability of Up Move (p):** `{prob_up:.4f}`")
                st.write(f"**Probability of Down Move (1-p):** `{prob_down:.4f}`")
                st.write(f"**Payoff (Up):** `${payoff_up:.2f}`")
                st.write(f"**Payoff (Down):** `${payoff_down:.2f}`")
