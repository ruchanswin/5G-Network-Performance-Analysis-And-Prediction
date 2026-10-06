import os
import sys

import pandas as pd
from os import listdir, path

# Make the repo root importable so the shared feature module is found
# regardless of the directory the script is started from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import add_derived_features

# Path to the directory containing the raw CSV files directly (no zip)
RAW_DATA_PATH = './RawData/'

# Get list of all CSV files in the raw data folder
csvs = [path.join(RAW_DATA_PATH, f) for f in listdir(RAW_DATA_PATH) if f.endswith('.csv')]

# Combine all CSV files into a single dataframe
raw_data = pd.concat([pd.read_csv(csv, low_memory=False) for csv in csvs])

# Features selected
features = ["Day", "Year", "Month", "Date", "hour", "min", "sec",
            'latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 
            'Transfer size', 'Transfer size-RX', 'Bitrate', 'Bitrate-RX', "send_data", 'square_id']

# Dropping rows with NaN
raw_data = raw_data[features].dropna()

# Remove rows with invalid values
raw_data = raw_data[(raw_data['latitude'] != 99.0) & (raw_data['longitude'] != 999.0)]
raw_data = raw_data[raw_data['speed'] != -1]
raw_data = raw_data[(raw_data['svr1'] != 1000) &
                    (raw_data['svr2'] != 1000) &
                    (raw_data['svr3'] != 1000) &
                    (raw_data['svr4'] != 1000)]

# Convert columns to int
raw_data['YEAR'] = raw_data['Year'].astype(int)
raw_data['MONTH'] = raw_data['Month'].astype(int)
raw_data['DATE'] = raw_data['Date'].astype(int)
raw_data['HOUR'] = raw_data['hour'].astype(int)
raw_data['MIN'] = raw_data['min'].astype(int)
raw_data['SEC'] = raw_data['sec'].astype(int)

raw_data = raw_data.drop(columns=['Year', 'Month', 'Date', 'hour', 'min', 'sec'])

# Rename raw metric columns to the canonical names, then derive the model
# features through the shared module (single source of truth).
raw_data = raw_data.rename(columns={
    'Transfer size': 'upload_transfer_size_mbytes',
    'Transfer size-RX': 'download_transfer_size_rx_mbytes',
    'Bitrate': 'upload_bitrate_mbits/sec',
    'Bitrate-RX': 'download_bitrate_rx_mbits/sec',
})
add_derived_features(raw_data, inplace=True)

raw_data['DATE'] = pd.to_datetime(raw_data['YEAR'].astype(str) + '-' + raw_data['MONTH'].astype(str) + '-' + raw_data['DATE'].astype(str), format='%Y-%m-%d').dt.date
raw_data['TIME'] = pd.to_datetime(raw_data['HOUR'].astype(str) + ':' + raw_data['MIN'].astype(str) + ':' + raw_data['SEC'].astype(str), format='%H:%M:%S').dt.time

# Selecting clean features
clean_features = ['DATE', 'TIME', 'Day', 'latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 
                  'average_latency', 'upload_transfer_size_mbytes', 'download_transfer_size_rx_mbytes', 'total_bandwidth',
                  'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec', 'total_throughput', 'send_data', 'square_id']

raw_data = raw_data[clean_features]

# Renaming columns for clarity
raw_data = raw_data.rename(columns={
    'send_data': 'application_data',
    'Day': 'DAY'
})
# Save the cleaned data to a new CSV file
clean_csv_path = './ProcessedData/clean_data.csv'
raw_data.to_csv(clean_csv_path, index=False)

print(f"Cleaned data saved to {clean_csv_path}")