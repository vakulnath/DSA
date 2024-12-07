import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the provided CSV file
file_path = '/Users/vakulnath/Desktop/files/iss.csv'
iss_data = pd.read_csv(file_path)

# Filter columns containing "Status" to identify those satellites that are "in contact"
status_columns = [col for col in iss_data.columns if 'Status' in col]

# Create a new column for the count of satellites in contact at each minute
iss_data['Contacts'] = iss_data[status_columns].apply(lambda x: (x == 'in contact').sum(), axis=1)

# Process data for specific satellites: Mro and Maven
iss_data['Mro Contact'] = iss_data['Status'].apply(lambda x: 1 if x == 'in contact' else 0)
iss_data['Maven Contact'] = iss_data['Status.2'].apply(lambda x: 1 if x == 'in contact' else 0)

# Sum of connections for Mro and Maven
iss_data['Mro_Maven_Contacts'] = iss_data['Mro Contact'] + iss_data['Maven Contact']

# Plotting the overlapped data for overall contacts and specific contacts with Mro and Maven
plt.figure(figsize=(15, 6))
plt.plot(iss_data['Time'], iss_data['Contacts'], label='Total Contacts', color='blue')
plt.plot(iss_data['Time'], iss_data['Mro_Maven_Contacts'], label='Mro + Maven Contacts', color='red')
plt.title('Contacts with Mars Satellites Over 24 Hours')
plt.xlabel('Time')
plt.ylabel('Number of Contacts')
plt.yticks(range(5))  # Adjusting y-ticks to accommodate the highest possible total contacts
plt.xticks(iss_data['Time'][::60], rotation=90)  # Setting x-ticks to show every hour
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

