import ccxt
import pandas as pd
import json
import os
import time
import signal
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from dotenv import load_dotenv
import argparse
import sys
import talib
import numpy as np

# ANSI escape codes for colors
GREEN = '\033[32m'
RED = '\033[31m'
YELLOW = '\033[33m'
CYAN = '\033[36m'
RESET = '\033[0m'

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv('BITFINEX_API_KEY')
API_SECRET = os.getenv('BITFINEX_API_SECRET')

# Initialize Exchange
exchange = ccxt.bitfinex({
    'apiKey': API_KEY,
    'secret': API_SECRET,
})

# Parameters
CAPITAL = 49  # Total capital in USDT
RISK_PERCENT = 0.02  # Risk per trade (2%)
LEVERAGE = 10
PAIRS_FILE = "top_pairs.json"  # File to store top pairs
FETCH_INTERVAL_HOURS = 24  # Update top pairs every 24 hours
CHECK_INTERVAL_SECONDS = 1  # Reduced from 60 to 30 seconds for more frequent checks

# Shutdown Flag
shutdown_flag = False

# Signal Handler for Graceful Shutdown
def signal_handler(sig, frame):
    global shutdown_flag
    print("\nShutdown signal received. Cleaning up...")
    shutdown_flag = True

# Attach Signal Handlers
signal.signal(signal.SIGINT, signal_handler)  # Handles Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handles termination signals

# Add this new function at the top level
def get_available_markets():
    try:
        markets = exchange.load_markets()
        return markets
    except Exception as e:
        print(f"Error loading markets: {e}")
        return {}

# Add this function after get_available_markets
def verify_pairs(pairs):
    try:
        markets = exchange.load_markets()
        verified_pairs = []
        for pair in pairs:
            if pair in markets:
                verified_pairs.append(pair)
            else:
                print(f"Warning: {pair} is not available on Bitfinex, skipping...")
        
        if not verified_pairs:  # If no pairs were verified, return default pairs
            print("No valid pairs found, using default pairs...")
            return ["tBTCUSD", "tETHUSD"]
            
        return verified_pairs
        
    except Exception as e:
        print(f"Error verifying pairs: {e}")
        return ["tBTCUSD", "tETHUSD"]  # Return default pairs on error

# Fetch Top 5 Pairs by Volume
def fetch_top_pairs():
    try:
        markets = get_available_markets()
        tickers = exchange.fetch_tickers()
        
        if not tickers:
            print("Error: No tickers received from exchange")
            return []
            
        sorted_tickers = sorted(
            [(k, v) for k, v in tickers.items() if v.get('quoteVolume') is not None],
            key=lambda x: x[1]['quoteVolume'],
            reverse=True
        )
        
        top_pairs = [ticker[0] for ticker in sorted_tickers if 'USD' in ticker[0]]
        
        if not top_pairs:
            return ["tBTCUSD", "tETHUSD", "tSOLUSD", "tLTCUSD", "tEOSUSD", 
                   "tADAUSD", "tMATICUSD", "tAVAXUSD", "tATOMUSD", "tUNIUSD"]
            
        return top_pairs[:10]
        
    except Exception as e:
        print(f"Error in fetch_top_pairs: {e}")
        return ["tBTCUSD", "tETHUSD", "tSOLUSD", "tLTCUSD", "tEOSUSD",
                "tADAUSD", "tMATICUSD", "tAVAXUSD", "tATOMUSD", "tUNIUSD"]

# Save Top Pairs to File
def save_top_pairs(pairs):
    data = {
        "pairs": pairs,
        "timestamp": datetime.now().isoformat(),
    }
    with open(PAIRS_FILE, "w") as f:
        json.dump(data, f)

# Load Top Pairs from File
def load_top_pairs():
    if os.path.exists(PAIRS_FILE):
        try:
            with open(PAIRS_FILE, "r") as f:
                data = json.load(f)
            last_updated = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - last_updated < timedelta(hours=FETCH_INTERVAL_HOURS):
                # Convert old format to new format if needed
                pairs = data["pairs"]
                if pairs and ':USD' in pairs[0]:  # Old format detected
                    print("Converting old pair format to new format...")
                    pairs = [p.replace(':', '/') for p in pairs]
                return pairs
        except Exception as e:
            print(f"Error loading pairs file: {e}")
    return None

# Fetch or Update Top Pairs
def get_top_pairs():
    try:
        pairs = load_top_pairs()
        if not pairs:  # Fetch if not cached or outdated
            print("No cached pairs found or cache expired, fetching new pairs...")
            pairs = fetch_top_pairs()
            if pairs:  # Only save if we got valid pairs
                pairs = verify_pairs(pairs)  # Verify pairs before saving
                save_top_pairs(pairs)
        return pairs or ["tBTCUSD"]
    except Exception as e:
        print(f"Error in get_top_pairs: {e}")
        return ["tBTCUSD"]

# Fetch Market Data
def fetch_data(pair, timeframe='1m', limit=200):
    try:
        timeframe_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '3h': '3h', '6h': '6h', '12h': '12h',
            '1d': '1D', '1w': '1W', '1M': '1M',
        }
        
        tf = timeframe_map.get(timeframe, '1h')
        
        # Ensure pair is in correct format
        if not pair.startswith('t'):
            pair = f"t{pair}"
        
        print(f"Fetching data for {pair}...")  # Debug line
        
        bars = exchange.fetch_ohlcv(
            symbol=pair,
            timeframe=tf,
            limit=limit
        )
        
        if not bars or len(bars) < 10:
            print(f"Warning: Insufficient data for {pair} (received {len(bars) if bars else 0} bars)")
            return None
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {pair}: {str(e)}")
        return None

# Calculate Indicators
def calculate_indicators(df):
    df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_200'] = EMAIndicator(df['close'], window=200).ema_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd_diff()  # Histogram for confirmation
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    return df

# Analyze Candlestick Patterns
def analyze_candlestick_patterns(df):
    try:
        # Convert data to numpy arrays as required by TA-Lib
        open_data = df['open'].astype(float).values
        high_data = df['high'].astype(float).values
        low_data = df['low'].astype(float).values
        close_data = df['close'].astype(float).values
        
        patterns = {}
        
        # Most reliable candlestick patterns
        patterns['ENGULFING'] = talib.CDLENGULFING(open_data, high_data, low_data, close_data)
        patterns['HAMMER'] = talib.CDLHAMMER(open_data, high_data, low_data, close_data)
        patterns['SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(open_data, high_data, low_data, close_data)
        patterns['MORNING_STAR'] = talib.CDLMORNINGSTAR(open_data, high_data, low_data, close_data, penetration=0)
        patterns['EVENING_STAR'] = talib.CDLEVENINGSTAR(open_data, high_data, low_data, close_data, penetration=0)
        patterns['DOJI'] = talib.CDLDOJI(open_data, high_data, low_data, close_data)
        patterns['THREE_WHITE_SOLDIERS'] = talib.CDL3WHITESOLDIERS(open_data, high_data, low_data, close_data)
        patterns['THREE_BLACK_CROWS'] = talib.CDL3BLACKCROWS(open_data, high_data, low_data, close_data)
        
        # Return only the most recent value for each pattern
        return {k: v[-1] if isinstance(v, np.ndarray) else v for k, v in patterns.items()}
        
    except Exception as e:
        print(f"Error in candlestick analysis: {e}")
        return {}

# Generate Signal
def generate_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Get candlestick patterns
    patterns = analyze_candlestick_patterns(df)
    
    # Count bullish and bearish patterns
    bullish_patterns = sum(1 for v in patterns.values() if v > 0)
    bearish_patterns = sum(1 for v in patterns.values() if v < 0)
    
    # Quick trend check using EMA50
    is_uptrend = latest['ema_50'] > prev['ema_50']
    is_downtrend = latest['ema_50'] < prev['ema_50']
    
    # More aggressive RSI levels
    oversold = latest['rsi'] < 40  # Changed from 30
    overbought = latest['rsi'] > 60  # Changed from 70
    
    # MACD as secondary confirmation
    macd_bullish = latest['macd'] > prev['macd']  # Looking for increasing MACD
    macd_bearish = latest['macd'] < prev['macd']  # Looking for decreasing MACD
    
    # Generate signals with more lenient conditions
    if bullish_patterns >= 1:  # If any bullish pattern is found
        if (is_uptrend or oversold) and macd_bullish:  # Need only one of trend/RSI + MACD
            return 'buy'
    elif bearish_patterns >= 1:  # If any bearish pattern is found
        if (is_downtrend or overbought) and macd_bearish:  # Need only one of trend/RSI + MACD
            return 'sell'
    
    # Additional signals based on strong RSI moves
    if oversold and macd_bullish:  # RSI oversold + MACD turning up
        return 'buy'
    elif overbought and macd_bearish:  # RSI overbought + MACD turning down
        return 'sell'
    
    return None

# Calculate Position Size
def calculate_position_size(balance, entry_price, stop_loss_price):
    risk = balance * RISK_PERCENT
    position_size = risk / abs(entry_price - stop_loss_price)
    return position_size

# Place Order
def place_order(pair, side, amount, test_mode=False):
    if test_mode:
        print(f"TEST MODE - Would place {side} order for {pair}: Amount: {amount}")
        return {"test": True, "side": side, "amount": amount, "pair": pair}
    
    if side == 'buy':
        order = exchange.create_market_buy_order(pair, amount)
    elif side == 'sell':
        order = exchange.create_market_sell_order(pair, amount)
    return order

# Close All Positions and Cancel Orders
def close_all_positions_and_orders():
    try:
        # Cancel all open orders
        open_orders = exchange.fetch_open_orders()
        for order in open_orders:
            exchange.cancel_order(order['id'], order['symbol'])
            print(f"Cancelled order: {order['id']} for {order['symbol']}")
        
        # Close all open positions
        balance = exchange.fetch_balance()
        for symbol, details in balance['total'].items():
            if details > 0 and symbol != 'USDT':  # Check for non-zero positions
                pair = f"{symbol}/USDT"
                print(f"Closing position for {pair}")
                exchange.create_market_sell_order(pair, details)
    except Exception as e:
        print(f"Error closing positions or orders: {e}")

# Trade Logic for Each Pair
def trade_pair(pair, balance_per_pair, test_mode=False):
    try:
        df = fetch_data(pair)
        if df is None or len(df) < 10:
            return
            
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df['close'].isnull().any():
            return
            
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Get all conditions
        patterns = analyze_candlestick_patterns(df)
        is_uptrend = latest['ema_50'] > prev['ema_50']
        is_downtrend = latest['ema_50'] < prev['ema_50']
        macd_bullish = latest['macd'] > prev['macd']
        macd_bearish = latest['macd'] < prev['macd']
        oversold = latest['rsi'] < 40
        overbought = latest['rsi'] > 60
        bullish_patterns = sum(1 for v in patterns.values() if v > 0)
        bearish_patterns = sum(1 for v in patterns.values() if v < 0)
        
        signal = generate_signal(df)
        price_change = ((latest['close'] - prev['close']) / prev['close']) * 100
        
        if test_mode:
            print(f"\n=== {pair} Analysis ===")
            print(f"Price: ${latest['close']:.2f} ({price_change:+.2f}%)")
            print(f"RSI: {latest['rsi']:.2f} ({'OVERSOLD' if oversold else 'OVERBOUGHT' if overbought else 'NEUTRAL'})")
            print(f"MACD: {latest['macd']:.4f} ({'Bullish' if macd_bullish else 'Bearish'})")
            
            # Show all patterns
            print("\nPatterns Detected:")
            active_patterns = {k: v for k, v in patterns.items() if v != 0}
            if active_patterns:
                for pattern, value in active_patterns.items():
                    direction = 'Bullish' if value > 0 else 'Bearish'
                    print(f"- {pattern}: {direction}")
            else:
                print("No patterns detected")
            
            print("\nTrend:", "UPTREND" if is_uptrend else "DOWNTREND")
            
            # Show all potential signals
            print("\nSignal Conditions:")
            
            # Pattern-based signals
            if bullish_patterns >= 1:
                status = "✓" if (is_uptrend or oversold) and macd_bullish else "×"
                color = GREEN if status == "✓" else RED
                print(f"{color}[{status}] Bullish Pattern + (Uptrend/Oversold) + MACD{RESET}")
            
            if bearish_patterns >= 1:
                status = "✓" if (is_downtrend or overbought) and macd_bearish else "×"
                color = GREEN if status == "✓" else RED
                print(f"{color}[{status}] Bearish Pattern + (Downtrend/Overbought) + MACD{RESET}")
            
            # RSI-based signals
            status = "✓" if oversold and macd_bullish else "×"
            color = GREEN if status == "✓" else RED
            print(f"{color}[{status}] RSI Oversold + MACD Bullish{RESET}")
            
            status = "✓" if overbought and macd_bearish else "×"
            color = GREEN if status == "✓" else RED
            print(f"{color}[{status}] RSI Overbought + MACD Bearish{RESET}")
            
            # Final Signal
            if signal:
                print(f"\n{GREEN}→ {signal.upper()} signal generated!{RESET}")
            else:
                print(f"\n{YELLOW}× No signal generated{RESET}")
        
        if signal:
            entry_price = latest['close']
            atr = latest['atr']
            
            # Tighter stop loss for faster trades
            stop_loss_price = entry_price - (1.5 * atr) if signal == 'buy' else entry_price + (1.5 * atr)
            take_profit_price = entry_price + (2 * abs(entry_price - stop_loss_price)) if signal == 'buy' else entry_price - (2 * abs(entry_price - stop_loss_price))
            
            position_size = calculate_position_size(balance_per_pair, entry_price, stop_loss_price)
            position_size = abs(position_size)
            
            if test_mode:
                print(f"\n{CYAN}Trade Details:{RESET}")
                print(f"Entry: ${entry_price:.2f}")
                print(f"Stop Loss: ${stop_loss_price:.2f}")
                print(f"Take Profit: ${take_profit_price:.2f}")
                print(f"Position Size: {position_size:.4f}")
            
            place_order(pair, signal, position_size, test_mode)
            
    except Exception as e:
        if test_mode:
            print(f"Error analyzing {pair}: {e}")

# Main Real-Time Execution
def main():
    args = parse_arguments()
    test_mode = args.test
    
    if test_mode:
        print("Running in TEST MODE - No real trades will be executed")
        
    # Delete the cached file to force fresh market data
    if os.path.exists(PAIRS_FILE):
        os.remove(PAIRS_FILE)
        print("Removed cached pairs file to fetch fresh market data")
    
    while not shutdown_flag:
        try:
            if args.pair:
                # Single pair mode - Use Bitfinex format
                pair = f"t{args.pair.upper()}USD"
                print(f"\n{'TEST MODE - ' if test_mode else ''}Monitoring single pair: {pair}")
                trade_pair(pair, CAPITAL, test_mode)
            else:
                # Multiple pairs mode
                top_pairs = get_top_pairs()
                if not top_pairs:
                    print("No pairs available for trading. Using default: tBTCUSD")
                    top_pairs = ["tBTCUSD"]
                
                balance_per_pair = CAPITAL / len(top_pairs)
                print(f"\n{'TEST MODE - ' if test_mode else ''}Monitoring {len(top_pairs)} Pairs: {top_pairs}")
                
                for pair in top_pairs:
                    if shutdown_flag:
                        break
                    trade_pair(pair, balance_per_pair, test_mode)
                    time.sleep(1)  # Add delay between pairs to avoid rate limits
            
            # Add a countdown timer for next check
            for i in range(CHECK_INTERVAL_SECONDS, 0, -1):
                if shutdown_flag:
                    break
                if i % 5 == 0 or i <= 3:  # Only show countdown every 5 seconds and last 3 seconds
                    print(f"\rNext update in {i}s... (Ctrl+C to exit)", end='')
                time.sleep(1)
            print("\n")
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            if test_mode:
                import traceback
                print("Stack trace:")
                traceback.print_exc()
            if not test_mode:
                close_all_positions_and_orders()
            time.sleep(CHECK_INTERVAL_SECONDS)
    
    # Cleanup on Shutdown
    if not test_mode:
        close_all_positions_and_orders()
    print(f"{'TEST MODE - ' if test_mode else ''}Bot shutdown completed.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('pair', nargs='?', help='Trading pair (e.g., BTC, ETH)', default=None)
    parser.add_argument('--test', action='store_true', help='Run in test mode without real trades')
    return parser.parse_args()

if __name__ == "__main__":
    main()
