import ccxt
import pandas as pd
import json
import os
import time
import signal
import traceback
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from dotenv import load_dotenv
import argparse
import sys
import talib
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr
from time import sleep
from collections import deque
from threading import Lock
import pickle
from pathlib import Path
from threading import Thread
from queue import Queue
import websocket

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

# Initialize Exchange for derivatives
exchange = ccxt.bitfinex({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {
        'defaultType': 'swap',  # Use derivatives/swap markets
        'adjustForTimeDifference': True,
        'createMarketBuyOrderRequiresPrice': False,
        'fetchOHLCVWarning': False,
    },
    'enableRateLimit': True,
    'timeout': 30000,
})

# Parameters
CAPITAL = 49  # Total capital in USDT
RISK_PERCENT = 0.02  # Risk per trade (2%)
LEVERAGE = 10  # Default leverage for derivatives
PAIRS_FILE = "top_pairs.json"  # File to store top pairs
FETCH_INTERVAL_HOURS = 6  # Update pairs every 6 hours
CHECK_INTERVAL_SECONDS = 1  # Check interval for trading

# Most liquid derivative pairs on Bitfinex
DERIVATIVE_PAIRS = {
    'BTC': 'BTC/USDT:USDT',
    'ETH': 'ETH/USDT:USDT',
    'SOL': 'SOL/USDT:USDT',
    'XRP': 'XRP/USDT:USDT',
    'DOGE': 'DOGE/USDT:USDT',
    'ADA': 'ADA/USDT:USDT',
    'AVAX': 'AVAX/USDT:USDT',
    'LINK': 'LINK/USDT:USDT',
    'DOT': 'DOT/USDT:USDT',
    'MATIC': 'MATIC/USDT:USDT'
}

# Default derivative pairs if others are not available
DEFAULT_PAIRS = ['BTC/USDT:USDT', 'ETH/USDT:USDT']

# Shutdown Flag
shutdown_flag = False

# Update signal handler to not close positions
def signal_handler(sig, frame):
    global shutdown_flag
    print("\nShutdown signal received. Cleaning up...")
    shutdown_flag = True
    try:
        # Only cleanup WebSocket connection
        if realtime_manager and realtime_manager.ws:
            realtime_manager.ws.close()
        sys.exit(0)
    except Exception as e:
        print(f"Error during shutdown: {e}")
        sys.exit(1)

# Attach Signal Handlers
signal.signal(signal.SIGINT, signal_handler)  # Handles Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handles termination signals

# Add this new function at the top level
def get_available_markets():
    try:
        markets = exchange.load_markets()
        # Get both perpetual futures and spot markets
        derivative_markets = {
            symbol: data for symbol, data in markets.items() 
            if ':USDT' in symbol or (symbol.endswith('/USDT') and not symbol.startswith('TEST'))
        }
        return derivative_markets
    except Exception as e:
        print(f"Error loading markets: {e}")
        return {}

# Update verify_pairs function
def verify_pairs(pairs):
    try:
        markets = get_available_markets()
        verified_pairs = []
        
        # First try to get all derivative pairs
        derivative_markets = {k: v for k, v in markets.items() if ':USDT' in k}
        print(f"\nFound {len(derivative_markets)} derivative markets")
        
        for pair in pairs:
            # Convert any format to derivative format
            if ':USDT' not in pair:
                base = pair.split('/')[0] if '/' in pair else pair.replace('t', '').replace('USD', '').replace('USDT', '')
                pair = f"{base}/USDT:USDT"
            
            if pair in derivative_markets:
                if pair not in verified_pairs:  # Avoid duplicates
                    verified_pairs.append(pair)
            else:
                print(f"Warning: {pair} is not available on Bitfinex derivatives")
        
        if not verified_pairs:
            print("No valid derivative pairs found, using default pairs...")
            verified_defaults = [p for p in DEFAULT_PAIRS if p in derivative_markets]
            return verified_defaults if verified_defaults else ['BTC/USDT:USDT']
        
        return verified_pairs
        
    except Exception as e:
        print(f"Error verifying pairs: {e}")
        return ['BTC/USDT:USDT']

# Update fetch_top_pairs
def fetch_top_pairs():
    try:
        print("\nFetching top volume pairs...")
        markets = get_available_markets()
        
        # Get all derivative pairs
        derivative_markets = {k: v for k, v in markets.items() if ':USDT' in k}
        if not derivative_markets:
            print("No derivative markets found, using default pairs")
            return DEFAULT_PAIRS
            
        print(f"\nFound {len(derivative_markets)} derivative markets")
        
        # Fetch all tickers at once
        all_tickers = exchange.fetch_tickers()
        
        # Process volume data for derivative pairs
        volume_data = []
        for pair in derivative_markets.keys():
            if pair in all_tickers:
                ticker = all_tickers[pair]
                # Calculate USD volume from baseVolume and last price
                if ticker.get('baseVolume') and ticker.get('last'):
                    volume_usd = float(ticker['baseVolume']) * float(ticker['last'])
                    volume_data.append((pair, volume_usd))
        
        if not volume_data:
            print("No volume data available, using default pairs")
            return DEFAULT_PAIRS
        
        # Sort by volume
        sorted_pairs = sorted(volume_data, key=lambda x: x[1], reverse=True)
        
        # Get top 10 pairs
        top_pairs = [pair for pair, volume in sorted_pairs[:10]]
        
        if top_pairs:
            print("\nTop 10 pairs by 24h volume:")
            for i, (pair, volume) in enumerate(sorted_pairs[:10], 1):
                print(f"{i}. {pair}: ${volume:,.2f}")
            return top_pairs
        else:
            print("No valid pairs found, using default pairs")
            return DEFAULT_PAIRS
        
    except Exception as e:
        print(f"Error in fetch_top_pairs: {e}")
        print("Using default pairs due to error")
        return DEFAULT_PAIRS

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
            
            # Check if data is older than FETCH_INTERVAL_HOURS
            if datetime.now() - last_updated < timedelta(hours=FETCH_INTERVAL_HOURS):
                pairs = data["pairs"]
                print(f"\nUsing cached pairs (updated {last_updated.strftime('%Y-%m-%d %H:%M:%S')})")
                print("Current pairs:", pairs)
                return pairs
            else:
                print(f"\nPairs data is older than {FETCH_INTERVAL_HOURS} hours, fetching new data...")
                return None
        except Exception as e:
            print(f"Error loading pairs file: {e}")
    return None

# Update get_top_pairs
def get_top_pairs():
    try:
        # Force refresh of pairs
        print("\nFetching fresh list of pairs...")
        pairs = fetch_top_pairs()
        
        if not pairs:
            print("No pairs found, using default pairs")
            return DEFAULT_PAIRS
        
        # Save new pairs
        save_top_pairs(pairs)
        
        # Verify pairs are valid derivatives
        markets = get_available_markets()
        valid_pairs = [p for p in pairs if p in markets and ':USDT' in p]
        
        if not valid_pairs:
            print("No valid derivative pairs found, using default pairs")
            return DEFAULT_PAIRS
        
        print(f"\nTrading {len(valid_pairs)} pairs:")
        for pair in valid_pairs:
            print(f"- {pair}")
        
        return valid_pairs
        
    except Exception as e:
        print(f"Error in get_top_pairs: {e}")
        return DEFAULT_PAIRS

# Update rate limiting parameters to be more conservative
RATE_LIMIT = {
    'max_requests': 15,     # Reduced from 30 to 15 requests per window
    'time_window': 60,      # Keep 60 second window
    'delay_between_pairs': 5,  # Increased from 2 to 5 seconds
    'retry_delay': 30,      # Increased from 10 to 30 seconds
    'max_retries': 3,
    'base_delay': 5,       # Base delay between requests
}

# Add cache directory configuration
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

class DataCache:
    def __init__(self, timeframe='1m'):
        self.timeframe = timeframe
        self.data = {}
        self.last_update = {}
        self.load_cache()
        self.update_queue = set()  # Track pairs that need updates
        self.last_correlation_update = datetime.min
        self.correlation_matrix = None
    
    def get_cache_file(self, pair):
        """Get cache file path for a pair"""
        return CACHE_DIR / f"{pair.replace('/', '_')}_{self.timeframe}.pkl"
    
    def load_cache(self):
        """Load all cached data at startup"""
        try:
            for cache_file in CACHE_DIR.glob("*.pkl"):
                try:
                    pair = cache_file.stem.split('_')[0].replace('_', '/')
                    with open(cache_file, 'rb') as f:
                        self.data[pair] = pickle.load(f)
                    print(f"Loaded cached data for {pair}")
                except Exception as e:
                    print(f"Error loading cache for {cache_file}: {e}")
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def save_cache(self, pair):
        """Save cached data for a pair"""
        try:
            cache_file = self.get_cache_file(pair)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data[pair], f)
        except Exception as e:
            print(f"Error saving cache for {pair}: {e}")
    
    def get_data(self, pair, limit=200):
        """Get data from cache, update if needed"""
        current_time = datetime.now()
        
        # Initialize if pair not in cache
        if pair not in self.data:
            self.data[pair] = pd.DataFrame()
            self.last_update[pair] = datetime.min
            self.update_queue.add(pair)
            return self._fetch_initial_data(pair, limit)
        
        # Check if we need to update
        time_since_update = (current_time - self.last_update.get(pair, datetime.min)).total_seconds()
        if time_since_update >= 60:
            self.update_queue.add(pair)
        
        # Return current data while update is pending
        if len(self.data[pair]) >= limit:
            return self.data[pair].tail(limit).copy()
        
        # If insufficient data, fetch immediately
        return self._fetch_initial_data(pair, limit)
    
    def _fetch_initial_data(self, pair, limit):
        """Fetch initial historical data"""
        df = fetch_historical_data(pair, limit)
        if df is not None:
            self.data[pair] = df
            self.last_update[pair] = datetime.now()
            self.save_cache(pair)
        return df.copy() if df is not None else None
    
    def process_updates(self):
        """Process all pending updates efficiently"""
        if not self.update_queue:
            return
        
        current_time = datetime.now()
        pairs_to_update = list(self.update_queue)
        self.update_queue.clear()
        
        for pair in pairs_to_update:
            if pair not in self.data:
                continue
                
            latest_timestamp = self.data[pair]['timestamp'].max()
            new_data = fetch_recent_data(pair, latest_timestamp)
            
            if new_data is not None and not new_data.empty:
                self.data[pair] = pd.concat([self.data[pair], new_data])
                self.data[pair] = self.data[pair].drop_duplicates(subset=['timestamp'])
                self.data[pair] = self.data[pair].sort_values('timestamp').reset_index(drop=True)
                self.last_update[pair] = current_time
                self.save_cache(pair)
    
    def get_correlation_matrix(self, pairs, force_update=False):
        """Get correlation matrix with caching"""
        current_time = datetime.now()
        if (not force_update and 
            self.correlation_matrix is not None and 
            (current_time - self.last_correlation_update).total_seconds() < 300):  # 5 minutes cache
            return self.correlation_matrix
        
        self.correlation_matrix = calculate_correlation_matrix(pairs)
        self.last_correlation_update = current_time
        return self.correlation_matrix

# Initialize cache
data_cache = DataCache()

def fetch_historical_data(pair, limit=200):
    """Fetch historical data with rate limiting"""
    try:
        rate_limiter.wait_if_needed()
        ohlcv = exchange.fetch_ohlcv(
            symbol=pair,
            timeframe='1m',
            limit=limit,
            params={'sort': -1}
        )
        
        if not ohlcv or len(ohlcv) < 10:
            return None
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')
        return df
        
    except Exception as e:
        print(f"Error fetching historical data for {pair}: {e}")
        return None

def fetch_recent_data(pair, last_timestamp, max_retries=3):
    """Fetch only recent candles since last update"""
    for retry in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            
            # Convert timestamp to milliseconds for the API
            since = int(last_timestamp.timestamp() * 1000)
            
            ohlcv = exchange.fetch_ohlcv(
                symbol=pair,
                timeframe='1m',
                since=since,
                limit=10,  # Fetch fewer candles for updates
                params={'sort': -1}
            )
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[df['timestamp'] > last_timestamp]  # Filter out old candles
            
            return df
            
        except Exception as e:
            if "ratelimit" in str(e).lower() and retry < max_retries - 1:
                sleep_time = RATE_LIMIT['retry_delay'] * (2 ** retry)
                print(f"Rate limit hit, waiting {sleep_time} seconds before retry...")
                sleep(sleep_time)
                continue
            print(f"Error fetching recent data for {pair}: {e}")
            return None
    
    return None

# Update fetch_data to use cache
def fetch_data(pair, timeframe='1m', limit=200, max_retries=RATE_LIMIT['max_retries']):
    """Fetch market data using cache"""
    try:
        # Ensure correct pair format
        if not ':USDT' in pair and not pair.endswith('/USDT'):
            base = pair.replace('t', '').replace('USD', '').replace('USDT', '')
            pair = f"{base}/USDT:USDT"
        
        # Get data from cache
        df = data_cache.get_data(pair, limit)
        
        if df is None or len(df) < 10:
            print(f"Warning: Insufficient data for {pair}")
            return None
        
        # Verify data freshness
        latest_candle_time = df['timestamp'].max()
        time_diff = datetime.now() - latest_candle_time
        if time_diff > timedelta(minutes=5):
            print(f"Warning: Data might be stale. Latest candle is from {latest_candle_time}")
        
        return df
        
    except Exception as e:
        print(f"Error in fetch_data: {e}")
        return None

# Update RateLimiter class to be more conservative
class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()

    def can_make_request(self):
        with self.lock:
            now = datetime.now()
            # Remove requests older than the time window
            while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
                self.requests.popleft()
            
            # More conservative: leave buffer for unexpected requests
            if len(self.requests) < (self.max_requests * 0.8):  # Only use 80% of rate limit
                self.requests.append(now)
                return True
            return False

    def wait_if_needed(self):
        while not self.can_make_request():
            sleep(2)  # Increased from 1 to 2 second wait

# Initialize rate limiter
rate_limiter = RateLimiter(RATE_LIMIT['max_requests'], RATE_LIMIT['time_window'])

# Add new market analysis functions
def analyze_volume_profile(df, num_bins=20):
    """Analyze volume distribution across price levels"""
    try:
        # Calculate price bins and volume distribution
        price_range = df['close'].values
        volume_range = df['volume'].values
        
        # Create price bins
        bins = np.linspace(min(price_range), max(price_range), num_bins)
        
        # Calculate volume per price level
        volume_profile = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            mask = (price_range >= bins[i]) & (price_range < bins[i+1])
            volume_profile[i] = np.sum(volume_range[mask])
        
        # Find POC (Point of Control) - price level with highest volume
        poc_index = np.argmax(volume_profile)
        poc_price = (bins[poc_index] + bins[poc_index+1]) / 2
        
        # Calculate Value Area (70% of volume)
        total_volume = np.sum(volume_profile)
        value_area_volume = total_volume * 0.7
        
        # Find Value Area High and Low
        cumsum = 0
        vah = val = poc_price
        for i in range(len(volume_profile)):
            cumsum += volume_profile[i]
            if cumsum <= value_area_volume:
                val = (bins[i] + bins[i+1]) / 2
            if cumsum >= total_volume - value_area_volume:
                vah = (bins[i] + bins[i+1]) / 2
                break
        
        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'volume_profile': volume_profile,
            'price_bins': bins
        }
    except Exception as e:
        print(f"Error in volume profile analysis: {e}")
        return None

def get_funding_rate(pair):
    """Fetch and analyze funding rate"""
    try:
        # Fetch funding rate from Bitfinex
        funding_info = exchange.fetch_funding_rate(pair)
        
        if funding_info:
            current_rate = funding_info.get('fundingRate', 0)
            predicted_rate = funding_info.get('predictedFundingRate', 0)
            next_funding_time = funding_info.get('nextFundingTime')
            
            # Analyze funding rate
            funding_signal = 'neutral'
            if current_rate < -0.01:  # -1% funding rate
                funding_signal = 'long'  # Good time to go long
            elif current_rate > 0.01:  # +1% funding rate
                funding_signal = 'short'  # Good time to go short
            
            return {
                'current_rate': current_rate,
                'predicted_rate': predicted_rate,
                'next_funding_time': next_funding_time,
                'signal': funding_signal
            }
    except Exception as e:
        print(f"Error fetching funding rate: {e}")
    return None

def detect_market_regime(df, window=20):
    """Detect current market regime using volatility and trend"""
    try:
        # Calculate volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(window).std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend strength
        ema_short = df['ema_50']
        ema_long = df['ema_200']
        trend_strength = (ema_short - ema_long) / ema_long
        
        # Calculate volume trend
        volume_sma = df['volume'].rolling(window).mean()
        volume_trend = df['volume'] / volume_sma
        
        # Use Gaussian Mixture Model to classify market regime
        features = np.column_stack([
            volatility.fillna(0),
            trend_strength.fillna(0),
            volume_trend.fillna(1)
        ])
        
        gmm = GaussianMixture(n_components=4, random_state=42)
        regimes = gmm.fit_predict(features)
        
        # Analyze latest regime
        latest_regime = regimes[-1]
        latest_vol = volatility.iloc[-1]
        latest_trend = trend_strength.iloc[-1]
        latest_vol_trend = volume_trend.iloc[-1]
        
        # Classify regime
        if latest_vol > volatility.mean() + volatility.std():
            if latest_trend > 0:
                regime_type = 'volatile_bullish'
            else:
                regime_type = 'volatile_bearish'
        else:
            if abs(latest_trend) < 0.02:  # 2% threshold
                regime_type = 'ranging'
            else:
                regime_type = 'trending'
        
        return {
            'regime': regime_type,
            'volatility': latest_vol,
            'trend_strength': latest_trend,
            'volume_trend': latest_vol_trend
        }
    except Exception as e:
        print(f"Error in market regime detection: {e}")
        return None

# Update calculate_indicators function
def calculate_indicators(df):
    try:
        # Existing indicators
        df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(df['close'], window=200).ema_indicator()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        macd = MACD(df['close'])
        df['macd'] = macd.macd_diff()
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Add volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Add volatility analysis
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Add momentum indicators
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['rate_of_change'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        return df
    except Exception as e:
        print(f"Error calculating indicators: {e}")
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

# Update generate_signal function with more sophisticated analysis
def generate_signal(df):
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Get market analysis
        volume_profile = analyze_volume_profile(df)
        market_regime = detect_market_regime(df)
        
        # Get candlestick patterns
        patterns = analyze_candlestick_patterns(df)
        bullish_patterns = sum(1 for v in patterns.values() if v > 0)
        bearish_patterns = sum(1 for v in patterns.values() if v < 0)
        
        # Trend analysis
        is_uptrend = latest['ema_50'] > prev['ema_50']
        is_downtrend = latest['ema_50'] < prev['ema_50']
        trend_strength = abs(latest['ema_50'] - latest['ema_200']) / latest['ema_200']
        
        # Volume analysis
        volume_increasing = latest['volume_ratio'] > 1.5  # 50% above average
        
        # Volatility analysis
        high_volatility = latest['volatility'] > df['volatility'].mean() + df['volatility'].std()
        
        # Momentum analysis
        strong_momentum = abs(latest['rate_of_change']) > 2.0  # 2% price change
        momentum_aligned = (latest['momentum'] > 0) == (latest['macd'] > 0)
        
        # RSI conditions with dynamic thresholds based on volatility
        rsi_threshold_adjust = 5 if high_volatility else 0
        oversold = latest['rsi'] < (40 + rsi_threshold_adjust)
        overbought = latest['rsi'] > (60 - rsi_threshold_adjust)
        
        # MACD analysis
        macd_bullish = latest['macd'] > prev['macd']
        macd_bearish = latest['macd'] < prev['macd']
        
        # Score-based signal generation
        long_score = 0
        short_score = 0
        
        # Trend scoring
        if is_uptrend:
            long_score += 1
        if is_downtrend:
            short_score += 1
        
        # Pattern scoring
        long_score += bullish_patterns
        short_score += bearish_patterns
        
        # Volume scoring
        if volume_increasing:
            if is_uptrend:
                long_score += 1
            if is_downtrend:
                short_score += 1
        
        # Momentum scoring
        if strong_momentum and momentum_aligned:
            if latest['momentum'] > 0:
                long_score += 1
            else:
                short_score += 1
        
        # RSI scoring
        if oversold and macd_bullish:
            long_score += 1
        if overbought and macd_bearish:
            short_score += 1
        
        # Market regime adjustments
        if market_regime:
            regime_type = market_regime['regime']
            if regime_type == 'volatile_bullish':
                long_score *= 1.2
            elif regime_type == 'volatile_bearish':
                short_score *= 1.2
            elif regime_type == 'ranging':
                long_score *= 0.8
                short_score *= 0.8
        
        # Volume profile analysis
        if volume_profile:
            current_price = latest['close']
            if current_price < volume_profile['val']:
                long_score += 0.5
            elif current_price > volume_profile['vah']:
                short_score += 0.5
        
        # Final signal determination with higher threshold for volatile markets
        signal_threshold = 3 if high_volatility else 2
        
        if long_score > signal_threshold and long_score > short_score:
            return {
                'signal': 'buy',
                'strength': long_score / signal_threshold,
                'regime': market_regime['regime'] if market_regime else 'unknown'
            }
        elif short_score > signal_threshold and short_score > long_score:
            return {
                'signal': 'sell',
                'strength': short_score / signal_threshold,
                'regime': market_regime['regime'] if market_regime else 'unknown'
            }
        
        return None
        
    except Exception as e:
        print(f"Error generating signal: {e}")
        return None

def calculate_entry_exit_points(df, signal_info, atr_multiplier=1.5):
    """Calculate dynamic entry and exit points based on market conditions"""
    try:
        latest = df.iloc[-1]
        atr = latest['atr']
        
        if atr == 0 or pd.isna(atr):
            print("Warning: ATR is zero or invalid, using price percentage for stops")
            atr = latest['close'] * 0.01  # Use 1% of price as fallback
        
        # Get market regime
        market_regime = detect_market_regime(df)
        regime_type = market_regime['regime'] if market_regime else 'unknown'
        volatility = market_regime['volatility'] if market_regime else 1.0
        
        # Adjust ATR multiplier based on market regime and signal strength
        if regime_type in ['volatile_bullish', 'volatile_bearish']:
            atr_multiplier *= 1.5
        elif regime_type == 'ranging':
            atr_multiplier *= 0.8
        
        # Further adjust based on signal strength
        if signal_info and 'strength' in signal_info:
            atr_multiplier *= (1 + (signal_info['strength'] - 1) * 0.2)
        
        # Get recent price action
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        entry_price = latest['close']
        
        # Calculate stop loss and take profit distances
        stop_distance = atr * atr_multiplier
        tp_distance = stop_distance * 3  # 3:1 reward-risk ratio
        
        if signal_info['signal'] == 'buy':
            # For buy signals
            stop_loss = max(entry_price - stop_distance, recent_low * 0.99)  # Don't set stop below recent low
            take_profit = min(entry_price + tp_distance, recent_high * 1.02)  # Allow slightly higher TP for 3:1 ratio
        else:
            # For sell signals
            stop_loss = min(entry_price + stop_distance, recent_high * 1.02)  # Allow slightly higher stop for 3:1 ratio
            take_profit = max(entry_price - tp_distance, recent_low * 0.99)  # Don't set TP too far below recent low
        
        # Verify the prices make sense
        if signal_info['signal'] == 'buy':
            if not (stop_loss < entry_price < take_profit):
                print("Warning: Price levels invalid for buy signal")
                return None
        else:
            if not (take_profit < entry_price < stop_loss):
                print("Warning: Price levels invalid for sell signal")
                return None
        
        # Calculate reward/risk ratio
        rr_ratio = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        if rr_ratio < 2.8:  # Minimum 2.8:1 reward/risk ratio (allowing some flexibility from 3:1)
            print(f"Warning: Reward/Risk ratio too low: {rr_ratio:.2f}")
            return None
        
        return {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr_multiplier': atr_multiplier,
            'rr_ratio': rr_ratio
        }
        
    except Exception as e:
        print(f"Error calculating entry/exit points: {e}")
        return None

def calculate_correlation_matrix(pairs):
    """Calculate correlation matrix for all trading pairs"""
    try:
        price_data = {}
        # Fetch data for all pairs
        for pair in pairs:
            df = fetch_data(pair, limit=100)  # Use last 100 candles
            if df is not None:
                price_data[pair] = df['close'].pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = {}
        for pair1 in pairs:
            correlation_matrix[pair1] = {}
            for pair2 in pairs:
                if pair1 in price_data and pair2 in price_data:
                    corr, _ = pearsonr(price_data[pair1], price_data[pair2])
                    correlation_matrix[pair1][pair2] = corr
        
        return correlation_matrix
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None

def adjust_position_for_correlation(pair, position_size, all_pairs, open_positions):
    """Adjust position size based on correlations with open positions"""
    try:
        # Get correlation matrix
        correlations = calculate_correlation_matrix(all_pairs)
        if not correlations:
            return position_size
        
        # Calculate correlation-based adjustment
        total_correlation = 0
        num_correlated = 0
        
        for open_pair, pos_info in open_positions.items():
            if open_pair != pair and open_pair in correlations.get(pair, {}):
                corr = abs(correlations[pair][open_pair])
                if corr > 0.5:  # Consider pairs with >0.5 correlation
                    total_correlation += corr
                    num_correlated += 1
        
        if num_correlated > 0:
            # Reduce position size based on average correlation
            avg_correlation = total_correlation / num_correlated
            correlation_factor = 1 - (avg_correlation * 0.5)  # Reduce by up to 50%
            return position_size * correlation_factor
        
        return position_size
        
    except Exception as e:
        print(f"Error adjusting position for correlation: {e}")
        return position_size

def calculate_dynamic_risk_percent(market_regime, signal_strength):
    """Calculate dynamic risk percentage based on market conditions"""
    try:
        base_risk = RISK_PERCENT
        
        # Adjust risk based on market regime
        regime_type = market_regime.get('regime', 'unknown')
        regime_multiplier = {
            'volatile_bullish': 0.8,   # Reduce risk in volatile markets
            'volatile_bearish': 0.8,
            'ranging': 0.9,            # Slightly reduce risk in ranging markets
            'trending': 1.1,           # Increase risk in trending markets
            'unknown': 1.0
        }.get(regime_type, 1.0)
        
        # Adjust risk based on signal strength
        strength_multiplier = min(1.2, max(0.8, signal_strength))
        
        # Adjust risk based on volatility
        volatility = market_regime.get('volatility', 1.0)
        vol_multiplier = 1.0
        if volatility > 1.5:  # High volatility
            vol_multiplier = 0.8
        elif volatility < 0.5:  # Low volatility
            vol_multiplier = 1.2
        
        # Calculate final risk percentage
        adjusted_risk = base_risk * regime_multiplier * strength_multiplier * vol_multiplier
        
        # Cap maximum risk
        return min(adjusted_risk, RISK_PERCENT * 1.5)
        
    except Exception as e:
        print(f"Error calculating dynamic risk: {e}")
        return RISK_PERCENT

def calculate_position_size(balance, entry_price, stop_loss_price, pair, signal_info=None, open_positions=None, all_pairs=None):
    """Calculate position size with advanced risk management"""
    try:
        # Get market regime from signal info
        market_regime = {'regime': signal_info.get('regime', 'unknown')} if signal_info else {}
        signal_strength = signal_info.get('strength', 1.0) if signal_info else 1.0
        
        # Calculate dynamic risk percentage
        risk_percent = calculate_dynamic_risk_percent(market_regime, signal_strength)
        
        # Calculate risk amount in USDT
        risk_amount = balance * risk_percent
        
        # Calculate price difference for stop loss
        price_diff_percent = abs(entry_price - stop_loss_price) / entry_price
        
        # Calculate maximum position size considering leverage
        max_position_value = balance * LEVERAGE  # Maximum position value with leverage
        
        # Calculate position size based on risk
        position_value = risk_amount / price_diff_percent  # Value of position in USDT
        
        # Cap position value at maximum allowed
        position_value = min(position_value, max_position_value)
        
        # Convert to actual position size
        position_size = position_value / entry_price
        
        # Adjust for correlations if we have the necessary information
        if open_positions and all_pairs:
            position_size = adjust_position_for_correlation(pair, position_size, all_pairs, open_positions)
        
        # Round to appropriate precision
        position_size = round(position_size, 4)
        
        # Log position sizing details
        print(f"\nPosition Sizing Details:")
        print(f"Risk %: {risk_percent*100:.2f}%")
        print(f"Position Size: {position_size:.4f}")
        print(f"Position Value: ${position_value:.2f}")
        print(f"Max Leverage Value: ${max_position_value:.2f}")
        print(f"Using {LEVERAGE}x leverage")
        
        # Verify position size is reasonable
        if position_value > max_position_value:
            print(f"Warning: Position value exceeds maximum allowed with leverage")
            return 0
        
        if position_size * entry_price < 1:  # Minimum position value of 1 USDT
            print(f"Warning: Position value too small")
            return 0
            
        return position_size
        
    except Exception as e:
        print(f"Error calculating position size: {e}")
        return 0

# Update place_order function with correct Bitfinex endpoints
def place_order(pair, side, amount, test_mode=False, trade_points=None):
    try:
        if test_mode:
            print(f"TEST MODE - Would place {side} order for {pair}:")
            print(f"- Entry: Market order, Amount: {amount}")
            if trade_points:
                print(f"- Stop Loss: ${trade_points['stop_loss']:.2f}")
                print(f"- Take Profit: ${trade_points['take_profit']:.2f}")
            return {"test": True, "side": side, "amount": amount, "pair": pair}
        
        # Ensure we're in derivatives mode
        exchange.options['defaultType'] = 'swap'
        
        # Convert pair to Bitfinex derivative format
        formatted_pair = pair
        if ':USDT' in pair:
            base = pair.split('/')[0]
            formatted_pair = f"t{base}F0:USTF0"
        
        print(f"Using formatted pair: {formatted_pair}")
        
        # Set up order parameters
        params = {
            'reduceOnly': False,  # Allow position increase
            'leverage': LEVERAGE,  # Set leverage in params
        }
        
        # Try to set leverage
        try:
            exchange.private_post_derivatives_position_leverage({
                'symbol': formatted_pair,
                'leverage': str(LEVERAGE)
            })
        except Exception as e1:
            try:
                exchange.private_post_position_leverage({
                    'symbol': formatted_pair,
                    'lev': str(LEVERAGE)
                })
            except Exception as e2:
                print(f"Warning: Could not set leverage via API, using params instead")
        
        print(f"Leverage set to {LEVERAGE}x for {formatted_pair}")
        
        orders = []
        try:
            # Place main entry order
            if side == 'buy':
                entry_order = exchange.create_market_buy_order(
                    formatted_pair,
                    amount,
                    params=params
                )
            else:
                entry_order = exchange.create_market_sell_order(
                    formatted_pair,
                    amount,
                    params=params
                )
            orders.append(entry_order)
            print(f"Entry order placed successfully: {entry_order['id']}")
            
            # Place stop loss and take profit orders if trade points are provided
            if trade_points:
                # Format prices to string with correct precision
                stop_price = float(f"{trade_points['stop_loss']:.2f}")
                take_profit_price = float(f"{trade_points['take_profit']:.2f}")
                
                # Place stop loss
                sl_side = 'sell' if side == 'buy' else 'buy'
                sl_params = {
                    'reduceOnly': True,
                    'triggerPrice': stop_price,
                    'stopLoss': {
                        'price': stop_price
                    }
                }
                
                sl_order = exchange.create_order(
                    formatted_pair,
                    'STOP',  # Bitfinex stop order type
                    sl_side,
                    amount,
                    stop_price,
                    sl_params
                )
                orders.append(sl_order)
                print(f"Stop loss order placed: {sl_order['id']}")
                
                # Place take profit
                tp_side = 'sell' if side == 'buy' else 'buy'
                tp_params = {
                    'reduceOnly': True,
                    'postOnly': True
                }
                
                tp_order = exchange.create_order(
                    formatted_pair,
                    'LIMIT',  # Bitfinex limit order type
                    tp_side,
                    amount,
                    take_profit_price,
                    tp_params
                )
                orders.append(tp_order)
                print(f"Take profit order placed: {tp_order['id']}")
            
            return orders
            
        except Exception as e:
            if 'compliance' in str(e).lower():
                print(f"Error: Your account might be restricted from trading derivatives.")
                print("Please verify that:")
                print("1. Your account is verified for derivatives trading")
                print("2. Derivatives trading is allowed in your region")
                print("3. You have the required permissions")
            else:
                print(f"Error placing orders: {e}")
                if test_mode:
                    import traceback
                    traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Error in place_order: {e}")
        return None

# Close All Positions and Cancel Orders
def close_all_positions_and_orders():
    try:
        # Ensure we're in derivatives mode
        exchange.options['defaultType'] = 'swap'
        
        # Cancel all open orders first
        try:
            open_orders = exchange.fetch_open_orders()
            for order in open_orders:
                exchange.cancel_order(order['id'], order['symbol'])
                print(f"Cancelled order: {order['id']} for {order['symbol']}")
        except Exception as e:
            print(f"Error cancelling orders: {e}")
        
        # Close all derivative positions
        try:
            positions = exchange.fetch_positions()
            for position in positions:
                if float(position['contracts']) != 0:  # Check for non-zero positions
                    symbol = position['symbol']
                    amount = abs(float(position['contracts']))
                    side = 'sell' if float(position['contracts']) > 0 else 'buy'
                    print(f"Closing {side} position for {symbol}: {amount} contracts")
                    
                    params = {
                        'type': 'MARKET',
                        'reduceOnly': True  # Ensure we're only closing positions
                    }
                    
                    try:
                        if side == 'sell':
                            exchange.create_market_sell_order(symbol, amount, params)
                        else:
                            exchange.create_market_buy_order(symbol, amount, params)
                        print(f"Position closed for {symbol}")
                    except Exception as e:
                        print(f"Error closing position for {symbol}: {e}")
        except Exception as e:
            print(f"Error fetching positions: {e}")
                    
    except Exception as e:
        print(f"Error in close_all_positions_and_orders: {e}")

# Trade Logic for Each Pair
def trade_pair(pair, balance_per_pair, test_mode=False, open_positions=None, all_pairs=None):
    try:
        print(f"\nAnalyzing {pair}...")
        
        # Add delay between pairs
        sleep(RATE_LIMIT['delay_between_pairs'])
        
        # Fetch and prepare data
        print(f"Fetching data for {pair}...")
        df = fetch_data(pair)
        if df is None or len(df) < 10:
            print(f"Insufficient data for {pair}, skipping...")
            return
        
        print(f"Processing data for {pair}...")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df['close'].isnull().any():
            print(f"Invalid data for {pair}, skipping...")
            return
        
        # Calculate indicators and get market analysis
        print(f"Calculating indicators for {pair}...")
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Get funding rate if available
        if ':USDT' in pair:
            print(f"Fetching funding rate for {pair}...")
            funding_info = get_funding_rate(pair)
        else:
            funding_info = None
        
        # Generate trading signal with market regime detection
        print(f"Generating signal for {pair}...")
        signal_info = generate_signal(df)
        
        # Get market regime
        print(f"Detecting market regime for {pair}...")
        market_regime = detect_market_regime(df)
        
        # Get volume profile
        print(f"Analyzing volume profile for {pair}...")
        volume_profile = analyze_volume_profile(df)
        
        if test_mode:
            print(f"\n=== {pair} Analysis ===")
            print(f"Price: ${latest['close']:.2f}")
            print(f"Market Regime: {market_regime['regime'] if market_regime else 'unknown'}")
            print(f"Volatility: {market_regime['volatility']:.2%}" if market_regime else "Volatility: unknown")
            
            if funding_info:
                print(f"\nFunding Rate: {funding_info['current_rate']:.4%}")
                print(f"Predicted Rate: {funding_info['predicted_rate']:.4%}")
                print(f"Funding Signal: {funding_info['signal']}")
            
            if volume_profile:
                print(f"\nVolume Profile:")
                print(f"Point of Control: ${volume_profile['poc']:.2f}")
                print(f"Value Area High: ${volume_profile['vah']:.2f}")
                print(f"Value Area Low: ${volume_profile['val']:.2f}")
            
            print(f"\nTechnical Indicators:")
            print(f"RSI: {latest['rsi']:.2f}")
            print(f"MACD: {latest['macd']:.4f}")
            print(f"Volatility: {latest['volatility']:.2%}")
            print(f"Volume Ratio: {latest['volume_ratio']:.2f}x average")
        
        if signal_info:
            signal = signal_info['signal']
            print(f"\nSignal detected for {pair}: {signal}")
            
            # Check if funding rate aligns with signal
            if funding_info and funding_info['signal'] != 'neutral':
                if funding_info['signal'] != signal:
                    print(f"Funding rate signal mismatch for {pair}, skipping trade...")
                    if test_mode:
                        print(f"\nSkipping trade: Funding rate signal ({funding_info['signal']}) doesn't align with technical signal ({signal})")
                    return
            
            # Calculate entry and exit points
            print(f"Calculating trade points for {pair}...")
            trade_points = calculate_entry_exit_points(df, signal_info)
            if not trade_points:
                print(f"Could not calculate trade points for {pair}, skipping...")
                return
            
            entry_price = trade_points['entry']
            stop_loss_price = trade_points['stop_loss']
            take_profit_price = trade_points['take_profit']
            
            # Calculate position size with all risk factors
            print(f"Calculating position size for {pair}...")
            position_size = calculate_position_size(
                balance_per_pair,
                entry_price,
                stop_loss_price,
                pair,
                signal_info,
                open_positions,
                all_pairs
            )
            
            if test_mode:
                print(f"\n{GREEN if signal == 'buy' else RED}→ {signal.upper()} signal generated!{RESET}")
                print(f"\nSignal Strength: {signal_info['strength']:.2f}")
                print(f"Market Regime: {signal_info['regime']}")
                
                print(f"\n{CYAN}Trade Details:{RESET}")
                print(f"Entry: ${entry_price:.2f}")
                print(f"Stop Loss: ${stop_loss_price:.2f}")
                print(f"Take Profit: ${take_profit_price:.2f}")
                print(f"Position Size: {position_size:.4f}")
                print(f"Risk/Reward: {abs(take_profit_price - entry_price) / abs(stop_loss_price - entry_price):.2f}")
            
            # Place the order
            if position_size > 0:
                print(f"Placing order for {pair}...")
                place_order(pair, signal, position_size, test_mode, trade_points)
            else:
                print(f"Position size too small for {pair}, skipping trade...")
        else:
            print(f"No signal generated for {pair}")
            if test_mode:
                print(f"\n{YELLOW}× No signal generated{RESET}")
        
    except Exception as e:
        print(f"Error processing {pair}: {str(e)}")
        if test_mode:
            print(f"Error analyzing {pair}: {e}")
            import traceback
            traceback.print_exc()

# Add WebSocket support for real-time data
class RealtimeDataManager:
    def __init__(self):
        self.subscriptions = {}
        self.data_queues = {}
        self.ws = None
        self.running = False
        self.last_update = {}
        self.orderbook_data = {}
        self.trade_data = {}
        
    def start(self):
        """Start WebSocket connection"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if isinstance(data, list):
                    channel = data[1]
                    if channel == 'ticker':
                        self._handle_ticker(data[2])
                    elif channel == 'trades':
                        self._handle_trade(data[2])
                    elif channel == 'book':
                        self._handle_orderbook(data[2])
            except Exception as e:
                print(f"Error handling websocket message: {e}")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
            if self.running:
                self._reconnect()

        def on_open(ws):
            print("WebSocket connection opened")
            self._subscribe_all()

        self.ws = websocket.WebSocketApp(
            "wss://api-pub.bitfinex.com/ws/2",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.running = True
        Thread(target=self.ws.run_forever).start()
    
    def _subscribe_all(self):
        """Subscribe to all required channels"""
        for pair in self.subscriptions:
            # Subscribe to ticker
            self.ws.send(json.dumps({
                "event": "subscribe",
                "channel": "ticker",
                "symbol": pair
            }))
            # Subscribe to trades
            self.ws.send(json.dumps({
                "event": "subscribe",
                "channel": "trades",
                "symbol": pair
            }))
            # Subscribe to order book
            self.ws.send(json.dumps({
                "event": "subscribe",
                "channel": "book",
                "symbol": pair,
                "prec": "P0",
                "freq": "F0",
                "len": "25"
            }))

    def _reconnect(self):
        """Reconnect WebSocket"""
        if self.ws:
            self.ws.close()
        self.start()

    def subscribe(self, pair):
        """Subscribe to a new trading pair"""
        if pair not in self.subscriptions:
            self.subscriptions[pair] = True
            self.data_queues[pair] = Queue()
            if self.ws and self.ws.sock and self.ws.sock.connected:
                self._subscribe_pair(pair)

    def _subscribe_pair(self, pair):
        """Subscribe to all channels for a pair"""
        for channel in ['ticker', 'trades', 'book']:
            self.ws.send(json.dumps({
                "event": "subscribe",
                "channel": channel,
                "symbol": pair
            }))

    def _handle_ticker(self, data):
        """Handle ticker updates"""
        pair = data[0]
        ticker = {
            'bid': float(data[1]),
            'ask': float(data[3]),
            'last': float(data[7]),
            'volume': float(data[8]),
            'high': float(data[9]),
            'low': float(data[10])
        }
        self.last_update[pair] = ticker
        self.data_queues[pair].put(('ticker', ticker))

    def _handle_trade(self, data):
        """Handle trade updates"""
        pair = data[0]
        trade = {
            'price': float(data[3]),
            'amount': float(data[2]),
            'time': data[1]
        }
        if pair not in self.trade_data:
            self.trade_data[pair] = []
        self.trade_data[pair].append(trade)
        self.data_queues[pair].put(('trade', trade))

    def _handle_orderbook(self, data):
        """Handle orderbook updates"""
        pair = data[0]
        if len(data) > 1:
            price = float(data[0])
            count = int(data[1])
            amount = float(data[2])
            
            if pair not in self.orderbook_data:
                self.orderbook_data[pair] = {'bids': {}, 'asks': {}}
            
            if amount > 0:
                self.orderbook_data[pair]['bids'][price] = amount
            elif amount < 0:
                self.orderbook_data[pair]['asks'][price] = abs(amount)
            elif count == 0:
                if price in self.orderbook_data[pair]['bids']:
                    del self.orderbook_data[pair]['bids'][price]
                if price in self.orderbook_data[pair]['asks']:
                    del self.orderbook_data[pair]['asks'][price]

    def get_latest_data(self, pair):
        """Get latest data for a pair"""
        return {
            'ticker': self.last_update.get(pair),
            'trades': self.trade_data.get(pair, [])[-100:],  # Last 100 trades
            'orderbook': self.orderbook_data.get(pair)
        }

# Initialize real-time data manager
realtime_manager = RealtimeDataManager()

# Update fetch_data to use real-time data when possible
def fetch_data(pair, timeframe='1m', limit=200, max_retries=RATE_LIMIT['max_retries']):
    """Fetch market data using real-time data when possible"""
    try:
        # Ensure correct pair format
        if not ':USDT' in pair and not pair.endswith('/USDT'):
            base = pair.replace('t', '').replace('USD', '').replace('USDT', '')
            pair = f"{base}/USDT:USDT"
        
        # Subscribe to real-time data if not already
        realtime_manager.subscribe(pair)
        
        # Get real-time data
        latest_data = realtime_manager.get_latest_data(pair)
        
        # Get cached data
        df = data_cache.get_data(pair, limit)
        
        if df is None or len(df) < 10:
            print(f"Warning: Insufficient data for {pair}")
            return None
        
        # Update latest candle with real-time data
        if latest_data['ticker']:
            latest_row = df.iloc[-1]
            current_time = pd.Timestamp.now()
            
            # Only update if we're in the same minute
            if current_time.minute == latest_row.timestamp.minute:
                df.loc[df.index[-1], 'close'] = latest_data['ticker']['last']
                df.loc[df.index[-1], 'high'] = max(latest_row.high, latest_data['ticker']['last'])
                df.loc[df.index[-1], 'low'] = min(latest_row.low, latest_data['ticker']['last'])
                
                # Update volume from trades
                new_volume = sum(trade['amount'] for trade in latest_data['trades']
                    if pd.Timestamp(trade['time']).minute == current_time.minute)
                if new_volume > 0:
                    df.loc[df.index[-1], 'volume'] = new_volume
        
        # Verify data freshness
        latest_candle_time = df['timestamp'].max()
        time_diff = datetime.now() - latest_candle_time
        if time_diff > timedelta(minutes=5):
            print(f"Warning: Data might be stale. Latest candle is from {latest_candle_time}")
        
        return df
        
    except Exception as e:
        print(f"Error in fetch_data: {e}")
        return None

# Start real-time data manager in main
def main():
    global test_mode  # Make test_mode global for signal handler
    args = parse_arguments()
    test_mode = args.test
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if test_mode:
        print("Running in TEST MODE - No real trades will be executed")
    
    try:
        # Start real-time data manager
        realtime_manager.start()
        print("Started real-time data manager")
        
        # Load markets at startup
        try:
            rate_limiter.wait_if_needed()
            markets = exchange.load_markets()
            print("Markets loaded successfully")
        except Exception as e:
            print(f"Error loading markets: {e}")
            return
        
        # Initialize trading pairs
        if args.pair:
            base = args.pair.upper().replace('USDT', '').replace('/', '')
            trading_pairs = [f"{base}/USDT:USDT"]
            if trading_pairs[0] not in markets:
                print(f"Error: {args.pair} is not available for trading")
                return
        else:
            trading_pairs = get_top_pairs()
            if not trading_pairs:
                print("No pairs available for trading. Using default perpetual pairs")
                trading_pairs = DEFAULT_PAIRS
        
        # Main trading loop
        while not shutdown_flag:
            try:
                # Process any pending updates in cache
                data_cache.process_updates()
                
                # Calculate balance per pair
                balance_per_pair = CAPITAL / len(trading_pairs)
                
                # Process all pairs
                for pair in trading_pairs:
                    if shutdown_flag:
                        break
                    
                    try:
                        trade_pair(pair, balance_per_pair, test_mode)
                    except Exception as e:
                        print(f"Error processing {pair}: {e}")
                    
                    # Check shutdown flag after each pair
                    if shutdown_flag:
                        break
                    
                    # Small delay between pairs to respect rate limits
                    sleep(0.5)
                
                # Check shutdown flag before next iteration
                if shutdown_flag:
                    break
            
            except Exception as e:
                print(f"Error in main loop: {e}")
                if test_mode:
                    traceback.print_exc()
                if not test_mode:
                    close_all_positions_and_orders()
                if shutdown_flag:
                    break
                sleep(RATE_LIMIT['retry_delay'])
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Cleaning up WebSocket...")
    finally:
        # Only cleanup WebSocket on shutdown
        if realtime_manager and realtime_manager.ws:
            realtime_manager.ws.close()
        print(f"{'TEST MODE - ' if test_mode else ''}Bot shutdown completed.")
        sys.exit(0)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('pair', nargs='?', help='Trading pair (e.g., BTC, ETH)', default=None)
    parser.add_argument('--test', action='store_true', help='Run in test mode without real trades')
    return parser.parse_args()

if __name__ == "__main__":
    main()
