import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
data = df['Close'].values.reshape(-1, 1)  # Use closing prices

# Step 3: Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences: X (past sequence_length days), y (next day)
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM (samples, timesteps, features)

# Split into train/test (80/20 split)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()


# Custom Dataset for DataLoader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = StockDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Step 4: Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out


model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the model
print("Training model...")
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 6: Evaluate on test data
model.eval()
with torch.no_grad():
    predicted = model(X_test).numpy()
    predicted = scaler.inverse_transform(predicted)  # Inverse scale
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
print(f'\nTest RMSE: {rmse:.2f} (lower is better)')

# Step 7: Predict future prices (using last sequence_length days from data)
last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
future_predictions = []
with torch.no_grad():
    for _ in range(prediction_days):
        pred = model(torch.from_numpy(last_sequence).float()).numpy()
        future_predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Step 8: Plot results
plt.figure(figsize=(14, 7))

# Plot training data
train_dates = df.index[:split + sequence_length]
plt.plot(train_dates, data[:split + sequence_length], label='Training Data')

# Plot test actual vs predicted
test_dates = df.index[split + sequence_length:]
plt.plot(test_dates, actual, label='Actual Test Prices')
plt.plot(test_dates, predicted, label='Predicted Test Prices')

# Plot future predictions
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=prediction_days)
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--')

plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

print("\nFuture Predictions:")
for i, pred in enumerate(future_predictions):
    print(f"Day {i + 1}: {pred[0]:.2f}")