import os
import sys

import pandas as pd
import numpy as np
from os import listdir

# Make the repo root importable so the shared feature module is found
# regardless of the directory the script is started from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import add_derived_features

DATASET_PATH = './RawData/'
csvs = [DATASET_PATH + f for f in listdir(DATASET_PATH) if f.endswith('.csv')]

# Combining all csv files into a single dataframe
raw_data = pd.concat([pd.read_csv(csv, low_memory=False) for csv in csvs])

# Features selected
features = ["Day", "Year", "Month", "Date", "hour", "min", "sec",
            'latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 
            'Transfer size', 'Transfer size-RX', 'Bitrate', 'Bitrate-RX', "send_data", 'square_id']

# Dropping rows with NaN
raw_data = raw_data[features].dropna()

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

raw_data['DATES'] = pd.to_datetime(raw_data['YEAR'].astype(str) + '-' + raw_data['MONTH'].astype(str) + '-' + raw_data['DATE'].astype(str), format='%Y-%m-%d').dt.date
raw_data['TIME'] = pd.to_datetime(raw_data['HOUR'].astype(str) + ':' + raw_data['MIN'].astype(str) + ':' + raw_data['SEC'].astype(str), format='%H:%M:%S').dt.time
raw_data['Convert_time'] = pd.to_datetime(raw_data['DATES'].astype(str) + ' ' + raw_data['TIME'].astype(str)).dt.strftime('%Y-%m-%d %H:%M:%S')

clean_features = ["Convert_time", "DATES", "TIME", "Day", "YEAR", "MONTH", "DATE", "HOUR", "MIN", "SEC",
            'latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 
            'upload_transfer_size_mbytes', 'download_transfer_size_rx_mbytes', 
            'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec', "send_data", 'square_id', 
            'total_throughput', 'total_bandwidth', 'average_latency']

raw_data = raw_data[clean_features]
raw_data = raw_data.rename(columns={
    'send_data': 'application_data',
    'Day': 'DAY'
})
raw_data.to_csv('./ProcessedData/clean_data_training.csv', index=False)
