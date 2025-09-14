import os
from binance.client import Client
import pandas as pd
from datetime import datetime
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Binance client (replace with your API key and secret)
api_key = 'qOiSsHlp3eHzmec6rQXlg3svtXzJS0lPLRUeHhhoW4jeSYlLcNO10CiCpnhDxXI3'
api_secret = 'Xn9Rh9ESSeCTeW1NBCf3AM8mhIHdYcf9zCeDv8QROkhcsi3OTL97DZI0XV0PMNrr'
client = Client(api_key, api_secret)

# Define pairs and timeframes
pairs = [
    "BNBUSDT", "XRPUSDT", "SOLUSDT", "QNTUSDT", "HBARUSDT", "BTCUSDT", "ETHUSDT",
    "DOGEUSDT", "VETUSDT", "ADAUSDT", "FILUSDT", "THETAUSDT", "UNIUSDT",
    "APEUSDT", "ARBUSDT", "TRXUSDT", "OPUSDT"
  ]
timeframes = ['1h']

# Define the start date (until 2020)
start_date = '1 Sep, 2024'

# Create a directory to store the data if not exists
output_dir = "binance_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to fetch and save historical data
def fetch_and_save(pair, timeframe, start_date):
    print(f"Fetching data for {pair} with timeframe {timeframe} since {start_date}...")
    
    # Fetch historical data from Binance
    klines = client.get_historical_klines(pair, timeframe, start_date)
    
    # Convert data into a DataFrame
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set timestamp as the index
    df.set_index('timestamp', inplace=True)
    
    # Save the DataFrame as CSV
    file_name = f"{pair}_{timeframe}.csv"
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")

# Loop through all pairs and timeframes
for pair in pairs:
    for timeframe in timeframes:
        fetch_and_save(pair, timeframe, start_date)

print("Data download complete.")
