import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import time
import json
import os

# --- Configuration Block ---
CONFIG = {
    "stock_symbol": "TUPRS",
    "start_date": "01.01.2022 01:00:00",
    "end_date": "02.10.2025 21:59:40",
    "sequence_length": 60,
    "prediction_days": 30,
    "features": ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20'],
    "target_feature": 'Close',
    "model_params": {
        "hidden_size": 50,
        "num_layers": 2,
        "dropout_prob": 0.2
    },
    "training_params": {
        "epochs": 500,
        "batch_size": 32,
        "learning_rate": 0.001
    }
}

# --- Utility Functions ---
def to_epoch(date_str):
    dt = datetime.strptime(date_str, '%d.%m.%Y %H:%M:%S')
    return int(time.mktime(dt.timetuple()))

def download_stock_data(stockName, startTime, endTime):
    start_epoch = to_epoch(startTime)
    end_epoch = to_epoch(endTime)
    url = f"https://api.doviz.com/api/v12/assets/{stockName}/archive?start={start_epoch}&end={end_epoch}"
    headers = {
        "Authorization": "Bearer 51d8f2f9347d626a7c804eb0343cc69253d649b49edbc0b98e5b2a6bddd077cb"
    }
    json_filename = f"{stockName}_data.json"

    # Use cached data if it exists to avoid repeated API calls
    if os.path.exists(json_filename):
        print(f"Loading cached data from {json_filename}")
        with open(json_filename, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Downloaded and saved {stockName} data to {json_filename}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None

def doviz_json_to_df(json_data):
    archive = json_data.get("data", {}).get("archive", [])
    if not archive:
        raise ValueError("No archive data found in JSON")
    records = [
        {"Date": datetime.fromtimestamp(e["update_date"]), "Open": e["open"], "High": e["highest"],
         "Low": e["lowest"], "Close": e["close"], "Volume": e["volume"]} for e in archive
    ]
    df = pd.DataFrame(records)
    df.set_index("Date", inplace=True)
    df = df.sort_index()
    return df

# --- PyTorch Dataset and Model ---
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0 and c0 are initialized to zero by default in PyTorch LSTM, so explicit initialization isn't required
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) # Take the output from the last time step
        out = self.fc(out)
        return out

def main():
    # --- Step 1: Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 2: Data Fetching and Preparation ---
    print(f"Fetching data for {CONFIG['stock_symbol']}...")
    json_data = download_stock_data(CONFIG['stock_symbol'], CONFIG['start_date'], CONFIG['end_date'])
    if not json_data:
        return
    df = doviz_json_to_df(json_data)

    # --- Step 3: Feature Engineering ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True) # Remove rows with NaN values created by rolling mean

    data = df[CONFIG['features']].values
    target_col_idx = CONFIG['features'].index(CONFIG['target_feature'])

    # --- Step 4: Data Scaling and Sequencing ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(CONFIG['sequence_length'], len(scaled_data)):
        X.append(scaled_data[i - CONFIG['sequence_length']:i])
        y.append(scaled_data[i, target_col_idx]) # Target is the 'Close' price
    X, y = np.array(X), np.array(y)

    # --- Step 5: Data Splitting (Train, Validation, Test) ---
    train_split_idx = int(0.7 * len(X))
    val_split_idx = int(0.85 * len(X))

    X_train, y_train = X[:train_split_idx], y[:train_split_idx]
    X_val, y_val = X[train_split_idx:val_split_idx], y[train_split_idx:val_split_idx]
    X_test, y_test = X[val_split_idx:], y[val_split_idx:]

    # Convert to PyTorch tensors and move to device
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['training_params']['batch_size'], shuffle=True)

    # --- Step 6: Model Initialization ---
    model_config = {
        "input_size": len(CONFIG['features']),
        **CONFIG['model_params']
    }
    model = LSTMModel(**model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['training_params']['learning_rate'])

    # --- Step 7: Model Training & Validation ---
    print("Training model...")
    best_val_loss = float('inf')
    model_path = f"{CONFIG['stock_symbol']}_best_model.pth"

    for epoch in range(CONFIG['training_params']['epochs']):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val.unsqueeze(1))

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{CONFIG["training_params"]["epochs"]}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            # print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    print(f"\nTraining complete. Best model saved to {model_path}")

    # --- Step 8: Evaluate on Test Data ---
    # Load the best performing model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(X_test).cpu().numpy()

    # We need to inverse transform the predictions.
    # The scaler expects an array of shape (n_samples, n_features).
    # We create a dummy array of this shape, fill our predicted values
    # into the 'Close' price column, and then inverse transform.
    dummy_array = np.zeros((len(predicted_scaled), len(CONFIG['features'])))
    dummy_array[:, target_col_idx] = predicted_scaled.flatten()
    predicted = scaler.inverse_transform(dummy_array)[:, target_col_idx]

    dummy_array_actual = np.zeros((len(y_test), len(CONFIG['features'])))
    dummy_array_actual[:, target_col_idx] = y_test.cpu().numpy().flatten()
    actual = scaler.inverse_transform(dummy_array_actual)[:, target_col_idx]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f'\nTest RMSE: {rmse:.2f} (lower is better)')

    # --- Step 9: Predict Future Prices ---
    future_predictions_scaled = []
    last_sequence = torch.from_numpy(scaled_data[-CONFIG['sequence_length']:]).unsqueeze(0).float().to(device)

    with torch.no_grad():
        for _ in range(CONFIG['prediction_days']):
            pred = model(last_sequence)
            future_predictions_scaled.append(pred.item())

            # Create the next sequence for prediction
            new_pred_row = last_sequence.cpu().numpy()[0, -1, :].copy()
            new_pred_row[target_col_idx] = pred.item()
            next_sequence_np = np.vstack([last_sequence.cpu().numpy()[0, 1:, :], new_pred_row])
            last_sequence = torch.from_numpy(next_sequence_np).unsqueeze(0).float().to(device)

    # Inverse transform future predictions
    dummy_array_future = np.zeros((len(future_predictions_scaled), len(CONFIG['features'])))
    dummy_array_future[:, target_col_idx] = np.array(future_predictions_scaled).flatten()
    future_predictions = scaler.inverse_transform(dummy_array_future)[:, target_col_idx]

    # --- Step 10: Plot Results ---
    plt.figure(figsize=(15, 8))

    # Create a single source of truth for the dates that align with our y-values
    # This accounts for dates removed by dropna() and sequence creation
    dates_for_plotting = df.index[CONFIG['sequence_length']:]

    # Split the dates using the same indices as our data
    train_dates = dates_for_plotting[:train_split_idx]
    # val_dates = dates_for_plotting[train_split_idx:val_split_idx] # We don't plot val preds, but it's here for clarity
    test_dates = dates_for_plotting[val_split_idx:]
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=CONFIG['prediction_days'])

    # Check to ensure the lengths match before plotting
    assert len(test_dates) == len(actual), "Mismatch between test_dates and actual values!"
    assert len(test_dates) == len(predicted), "Mismatch between test_dates and predicted values!"

    # Plot historical training data
    plt.plot(df.index, df[CONFIG['target_feature']], label='Historical Prices', color='gray', alpha=0.7)

    # Plot test actual vs predicted
    plt.plot(test_dates, actual, label='Actual Test Prices', color='blue', marker='.', linestyle='None')
    plt.plot(test_dates, predicted, label='Predicted Test Prices', color='red')

    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='green', linestyle='--')

    plt.title(f'{CONFIG["stock_symbol"]} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel(f'{CONFIG["target_feature"]} Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nFuture Predictions:")
    for i, pred in enumerate(future_predictions):
        print(f"Day {i + 1} ({future_dates[i].strftime('%Y-%m-%d')}): {pred:.2f}")

if __name__ == '__main__':
    main()