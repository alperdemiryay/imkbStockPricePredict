import yfinance as yf

# Step 1: Parameters (customize these)
stock_symbol = 'GOOGL'  # e.g., 'AAPL' for Apple, 'GOOGL' for Google
start_date = '2022-01-01'  # Historical data start
end_date = '2025-08-22'  # End date (use current date or earlier for training)
sequence_length = 60  # Number of past days to use for prediction
prediction_days = 30  # Number of future days to predict
epochs = 50  # Training epochs (increase for better results, but slower)
batch_size = 32  # Batch size for training

# Step 2: Fetch historical data
print(f"Fetching data for {stock_symbol}...")
df = yf.download(stock_symbol, start=start_date, end=end_date)
print("HEAD 10:")
print(df.head(10))
print()
print("TAIL:")
print(df.tail())
print()
print("INFO:")
print(df.info())
print()
print("DESCRIBE: ")
print(df.describe())
df.to_csv('yfinanceTestData.csv')