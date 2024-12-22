import ccxt
import json
from pprint import pprint

def debug_exchange():
    print("\n=== Exchange Debug Info ===")
    
    # Initialize exchange
    exchange = ccxt.bitfinex({
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True,
        },
        'enableRateLimit': True,
    })
    
    try:
        # Load markets
        print("\nLoading markets...")
        markets = exchange.load_markets()
        
        # Get derivative markets
        derivative_markets = {k: v for k, v in markets.items() if ':USDT' in k}
        print(f"\nFound {len(derivative_markets)} derivative markets")
        print("\nFirst 5 derivative pairs:")
        pprint(list(derivative_markets.keys())[:5])
        
        # Check BTC perpetual
        print("\nFetching BTC/USDT:USDT ticker...")
        btc_ticker = exchange.fetch_ticker('BTC/USDT:USDT')
        print(f"BTC Perpetual Price: ${btc_ticker['last']}")
        print("Full BTC ticker info:")
        pprint(btc_ticker)
        
        # Check ETH perpetual
        print("\nFetching ETH/USDT:USDT ticker...")
        eth_ticker = exchange.fetch_ticker('ETH/USDT:USDT')
        print(f"ETH Perpetual Price: ${eth_ticker['last']}")
        
        # Test OHLCV data
        print("\nFetching BTC/USDT:USDT OHLCV data...")
        ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', timeframe='1m', limit=5)
        print("Last 5 candles:")
        for candle in ohlcv:
            print(f"Time: {exchange.iso8601(candle[0])} Open: ${candle[1]} High: ${candle[2]} Low: ${candle[3]} Close: ${candle[4]} Volume: {candle[5]}")
            
    except Exception as e:
        print(f"\nError during debug: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_exchange() 