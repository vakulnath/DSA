import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to count satellites based on the 'Route' column
def count_satellites(route):
    if pd.isna(route) or route == '':
        return 0
    else:
        return len(route.split(' -> '))

# Load the data from a CSV file
data_path = '/Users/vakulnath/Desktop/Celestia/satellite_routes2.csv'  # Replace 'your_data.csv' with the path to your CSV file
satellite_data = pd.read_csv(data_path)

# Convert 'Start Time' and 'End Time' to datetime to ensure time calculations work
satellite_data['Start Time'] = pd.to_datetime(satellite_data['Start Time'], format='%H:%M')
satellite_data['End Time'] = pd.to_datetime(satellite_data['End Time'], format='%H:%M')

# Handle cases where 'End Time' might be the next day (if necessary)
satellite_data.loc[satellite_data['End Time'] < satellite_data['Start Time'], 'End Time'] += pd.Timedelta(days=1)

# Apply counting satellites to each route entry
satellite_data['Satellite Count'] = satellite_data['Route'].apply(count_satellites)

# Initialize a list for all time points
all_times_adjusted = np.zeros(24 * 60)  # 24 hours * 60 minutes

# Iterate over the dataframe and mark the number of satellites for each minute range
for _, row in satellite_data.iterrows():
    start_idx = row['Start Time'].hour * 60 + row['Start Time'].minute
    end_idx = row['End Time'].hour * 60 + row['End Time'].minute + 1  # Include the end minute
    all_times_adjusted[start_idx:end_idx] = row['Satellite Count']

# Generate the x-axis labels for each minute in a full day
full_day_time_labels = pd.date_range("00:00", "23:59", freq="T").strftime('%H:%M')

# Plotting the full day (24-hour) satellite visibility
plt.figure(figsize=(16, 6))
plt.plot(full_day_time_labels, all_times_adjusted, marker='', linestyle='-')  # Plotting every minute
plt.xticks(full_day_time_labels[::60], rotation=90)  # Every hour label for better visibility
plt.yticks(np.arange(0, 5, 1))  # Number of satellites visible
plt.xlabel('Time of Day')
plt.ylabel('Number of Satellites')
plt.title('Channels Allocted from MRO or MAVEN to Mars Satellites')
plt.grid(True)
plt.tight_layout()
plt.show()
