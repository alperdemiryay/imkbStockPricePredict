import requests
from datetime import datetime
import time
import matplotlib.pyplot as plt
from tabulate import tabulate
import json

# Inputs
stockTicker = 'TUPRS'
startTimeInHrFormat = '17.04.2025 21:00:00'  # Example input in dd.mm.yyyy HH:MM:SS
endTimeInHrFormat = '22.08.2025 21:59:40'  # Example input in dd.mm.yyyy HH:MM:SS

# Convert to epoch time
def to_epoch(date_str):
    dt = datetime.strptime(date_str, '%d.%m.%Y %H:%M:%S')
    return int(time.mktime(dt.timetuple()))

def download_stock_data(stockName, startTime, endTime):
    # Convert to epoch time
    start_epoch = to_epoch(startTime)
    end_epoch = to_epoch(endTime)

    # Construct URL
    url = f"https://api.doviz.com/api/v12/assets/{stockName}/archive?start={start_epoch}&end={end_epoch}"

    # Headers
    headers = {
        "Authorization": "Bearer 51d8f2f9347d626a7c804eb0343cc69253d649b49edbc0b98e5b2a6bddd077cb"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse JSON response
        data = response.json()
        # Save to file
        with open(f"{stockName}_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Saved {stockName} json data to {stockName}_data.json")

        if data.get("error"):
            print("Error in response:", data)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return "ERROR!"


# Make the API request
try:
    data = download_stock_data(stockTicker, startTimeInHrFormat, endTimeInHrFormat)
    if data.get("error"):
        print("Error in response:", data)
    else:
        # Extract archive data
        archive = data.get("data", {}).get("archive", [])

        if not archive:
            print("No archive data found in response.")
        else:
            # Prepare data for table and plot
            table_data = []
            dates = []
            close_prices = []

            for entry in archive:
                # Convert epoch to human-readable date
                date = datetime.fromtimestamp(entry["update_date"]).strftime('%Y-%m-%d %H:%M:%S')
                table_data.append([
                    date,
                    entry["open"],
                    entry["highest"],
                    entry["lowest"],
                    entry["close"],
                    entry["volume"]
                ])
                dates.append(date)
                close_prices.append(entry["close"])

            # Print human-readable table
            headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
            print("\nStock Data for", stockTicker)
            print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f"))

            # Create plot
            plt.figure(figsize=(10, 5))
            plt.plot(dates, close_prices, marker='o', linestyle='-', color='b', label='Close Price')
            plt.xlabel('Date')
            plt.ylabel('Close Price (TRY)')
            plt.title(f'Stock Closing Prices for {stockTicker}')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # Display the plot
            plt.show()

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")