import pandas as pd
from datetime import datetime, timedelta
from ortools.linear_solver import pywraplp

def format_time(time_str):
    # Parse the time string
    time_obj = datetime.strptime(time_str, '%H:%M')

    # Format the time based on the hour
    if time_obj.hour < 10:
        return time_obj.strftime('%-H:%M')
    else:
        return time_obj.strftime('%H:%M')

def find_accessible_satellites(data, requested_time):
    time = format_time(requested_time)
    time_rows = data[data['Time'] == time]

    satellite_links = []
    for _, row in time_rows.iterrows():
        for col in data.columns:
            if col.startswith('To') and row[col] and (f'Status{col[2:]}' in data.columns) and row[f'Status{col[2:]}'] == 'in contact':
                from_col = 'From' if col == 'To' else f'From{col[2:]}'
                distance_col = f'Distance{col[2:]}'
                duration_col = f'Duration{col[2:]}'
                
                satellite_links.append({
                    'Time': row['Time'],
                    'From': row[from_col],
                    'To': row[col],
                    'Distance': float(row[distance_col]),
                    'Status': row[f'Status{col[2:]}'],
                    'Duration': row[duration_col]
                })
    return pd.DataFrame(satellite_links).drop_duplicates()

def find_next_available(data, requested_time):
    time = format_time(requested_time)
    check_limit = 90
    time_rows = data[data['Time'] == time]
    future_links = []

    for _, row in time_rows.iterrows():
        for col in data.columns:
            if col.startswith('To') and row[col] and (f'Status{col[2:]}' in data.columns) and row[f'Status{col[2:]}'] == 'not in contact':
                from_col = 'From' if col == 'To' else f'From{col[2:]}'

                for i in range(1, check_limit + 1):
                    future_time = add_minutes_to_time(time, i)
                    future_rows = data[(data['Time'] == future_time) & (data[from_col] == row[from_col]) & (data[col] == row[col])]

                    if not future_rows.empty and future_rows.iloc[0][f'Status{col[2:]}'] == 'in contact':
                        distance_col = f'Distance{col[2:]}'
                        duration_col = f'Duration{col[2:]}'

                        future_links.append({
                            'Time': future_time,
                            'From': row[from_col],
                            'To': row[col],
                            'Distance': float(future_rows.iloc[0][distance_col]),
                            'Status': 'in contact',
                            'Duration': future_rows.iloc[0][duration_col]
                        })
                        break

    return pd.DataFrame(future_links).drop_duplicates()

def calculate_transmission_time(satellite_links, data_size_mbits):
    speed_of_light_kmps = 299792  # Speed of light in kilometers per second
    transmission_rate_mbps = 1  # Transmission rate of 1 Mbps

    # Calculate light travel time in seconds and then convert to minutes
    satellite_links['Light Travel Time (minutes)'] = (satellite_links['Distance'] / speed_of_light_kmps) / 60

    # Calculate data transmission time in minutes
    satellite_links['Data Transmission Time (minutes)'] = data_size_mbits / transmission_rate_mbps / 60

    # Total transmission time in minutes
    satellite_links['Total Transmission Time (minutes)'] = satellite_links['Light Travel Time (minutes)'] + satellite_links['Data Transmission Time (minutes)']

    return satellite_links

def add_minutes_to_time(time_str, minutes):
    time_format = '%H:%M'
    new_time = datetime.strptime(time_str, time_format) + timedelta(minutes=minutes)
    hour = new_time.hour
    if hour < 10:
        time_format = '%-H:%M'  # Remove leading zero for single digit hours
    else:
        time_format = '%H:%M'   # Retain leading zero for double digit hours
    return new_time.strftime(time_format)

def convert_to_datetime(time_str):
    # Convert time string into a datetime object
    return datetime.strptime(time_str, '%H:%M')

def calculate_duration_in_minutes(start_time, end_time):
    # Convert both times to datetime
    start_dt = convert_to_datetime(start_time)
    end_dt = convert_to_datetime(end_time)

    # Handle the case where end_time is past midnight
    if end_dt < start_dt:
        end_dt += timedelta(days=1)

    # Calculate the duration in minutes
    duration = (end_dt - start_dt).total_seconds() / 60
    return duration

def create_light_travel_time_matrix(requested_time, satellite_links):
    # Trim spaces from 'From' and 'To' columns
    satellite_links['From'] = satellite_links['From'].str.strip()
    satellite_links['To'] = satellite_links['To'].str.strip()

    # Now create the set of unique satellite names
    satellites = list(set(satellite_links['From']).union(set(satellite_links['To'])))

    # Initialize the matrix with '-1' (representing unavailable paths)
    matrix = pd.DataFrame((-1), index=satellites, columns=satellites)

    # Populate the matrix with light travel times
    for _, row in satellite_links.iterrows():
        from_sat = row['From']
        to_sat = row['To']
        light_travel_time = float((row['Total Transmission Time (minutes)']))
        expected_time = add_minutes_to_time(requested_time, light_travel_time)
        
        # Convert expected_time and row['Duration'] to duration in minutes
        duration_in_minutes = calculate_duration_in_minutes(requested_time, row['Duration'])

        if from_sat != to_sat:  # Avoid self-loop paths
            if light_travel_time <= duration_in_minutes:
                matrix.loc[from_sat, to_sat] = light_travel_time
                matrix.loc[to_sat, from_sat] = light_travel_time  
    return matrix

def mars_solver(light_travel_time_matrix, start_satellite):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None, None

    satellites = list(light_travel_time_matrix.index)
    num_satellites = len(satellites)

    # Map satellite names to indices for easier handling
    satellite_index = {name: idx for idx, name in enumerate(satellites)}

    if start_satellite not in satellite_index:
        return None, None

    start_index = satellite_index[start_satellite]

    # Decision variables: x[j] = 1 if moving from start_satellite to satellite j, 0 otherwise
    x = {}
    for j in range(num_satellites):
        if light_travel_time_matrix.iloc[start_index, j] != -1:  # Only consider valid connections
            x[j] = solver.BoolVar(f'x[{start_index},{j}]')

    # Constraint: Only one outgoing connection from the start satellite
    solver.Add(solver.Sum(x[j] for j in x) == 1)

    # Objective: Minimize travel time
    solver.Minimize(
        solver.Sum(
            light_travel_time_matrix.iloc[start_index, j] * x[j]
            for j in x
        )
    )
    # Solve the problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        for j in x:
            if x[j].solution_value() > 0.5:  # Check which connection is chosen
                next_satellite = satellites[j]
                travel_time = light_travel_time_matrix.iloc[start_index, j]
                return next_satellite, travel_time
    return None, None  # No valid solution found

def print_route(visited_nodes, times):
    print("\n--- Route Summary ---")
    for i in range(len(visited_nodes)):
        print(f"{i + 1}. {visited_nodes[i]} at {times[i]}")
    print("----------------------\n")

def intermediary_satellite(data, start, unvisited, requested_time, data_size_mbps, visited_nodes, times):
    # Find future available links
    future_links = find_next_available(data, requested_time)
    if not future_links.empty:
        # Calculate future transmission times for the links
        future_transmission_times = calculate_transmission_time(future_links, data_size_mbps)

        # Filter for links that connect to or from unvisited satellites
        future_valid_links = future_transmission_times[
            ((future_transmission_times['To'].isin(unvisited)) | 
            (future_transmission_times['From'].isin(unvisited))) &
            ((future_transmission_times['To'].isin(visited_nodes)) | 
            (future_transmission_times['From'].isin(visited_nodes)))
        ]


        if not future_valid_links.empty:

            # Optionally, sort to find the earliest or most optimal link
            future_valid_links.sort_values(by=['Total Transmission Time (minutes)'], inplace=True)
            best_link = future_valid_links.iloc[0]

            # Output best link for debugging
            print("Best future link to use:", best_link)

            # Determine whether to move to or from an unvisited satellite
            if best_link['From'] in unvisited:
                next_satellite = best_link['From']  # Moving from an unvisited satellite
                start_satellite = best_link['To']
            else:
                next_satellite = best_link['To']  # Moving to an unvisited satellite
                start_satellite = best_link['From']

            # Simulate moving to the next satellite at the future time
            future_time = best_link['Time']
            travel_time = best_link['Total Transmission Time (minutes)']

            # Update visited nodes and times
            if next_satellite not in visited_nodes:
                visited_nodes.append(next_satellite)
                times.append(future_time)

            print(f"Planning to move from {start_satellite} to {next_satellite} at {future_time}, which will take {travel_time} minutes.")
        else:
            print("No valid future links to or from unvisited satellites available at this time.")
            print_route(visited_nodes, times)
    else:
        print("No future links available at this time.")
        print_route(visited_nodes, times)

def main():
    import csv

    # Load Mars satellite data and process
    mars_data = pd.read_csv('/Users/vakulnath/Desktop/files/mars.csv')
    requested_time = '4:00'
    data_size_mbps = 100  # Example data packet size
    visited_nodes = []  # Store the visited satellites in order
    times = []  # Store the corresponding times

    # List of all satellites
    all_satellites = ['Mro', 'Maven', 'Odyssey', 'Express']

    # Find accessible satellites based on the current time
    mars_satellite_links = find_accessible_satellites(mars_data, requested_time)
    future_links = find_next_available(mars_data, requested_time)
    print("CHannels available in future:")
    print(future_links)

    if mars_satellite_links.empty:
        print(f"No accessible satellites at {requested_time}. Exiting.")
        return

    # Calculate transmission times
    mars_transmission_times = calculate_transmission_time(mars_satellite_links, data_size_mbps)
    light_travel_time_matrix = create_light_travel_time_matrix(requested_time, mars_transmission_times)

    # Determine the initial depot
    depots = [sat for sat in ['Mro', 'Maven'] if sat in light_travel_time_matrix.index]
    if depots:
        depot_scores = light_travel_time_matrix.loc[depots].apply(lambda row: (row > 0).sum(), axis=1)
        start_satellite = depot_scores.idxmax()  # Choose the depot with the most connections
        visited_nodes.append(start_satellite)
        times.append(requested_time)
    else:
        start_satellite = None

    # Loop until the end time or until all nodes are visited
    while requested_time != "23:59":
        # Find accessible satellites based on the current time
        mars_satellite_links = find_accessible_satellites(mars_data, requested_time)
        mars_transmission_times = calculate_transmission_time(mars_satellite_links, data_size_mbps)

        # Create the light travel time matrix for the current time
        light_travel_time_matrix = create_light_travel_time_matrix(requested_time, mars_transmission_times)
        # Solve for the cheapest next option for the given satellite
        next_satellite, travel_time = mars_solver(light_travel_time_matrix, start_satellite)

        # Check if no valid next satellite is found
        if not next_satellite:
            print("No valid next satellite found. Ending the process.")
            break
        # Ensure the next satellite is not already visited, unless all nodes are visited
        while next_satellite in visited_nodes and set(visited_nodes) != set(all_satellites):
            print(f"{next_satellite} has already been visited. Searching for an alternative...")
            light_travel_time_matrix.loc[start_satellite, next_satellite] = -1  # Invalidate this choice
            next_satellite, travel_time = mars_solver(light_travel_time_matrix, start_satellite)

            # If no valid next satellite is found, stop the process
            if not next_satellite:
                print("No valid alternatives found. Stopping the process.")
                unvisited = list(set(all_satellites) - set(visited_nodes))
                intermediary_satellite(mars_data, start_satellite, unvisited, requested_time, data_size_mbps, visited_nodes, times)
                return

        # if all satellites are visited, break the loop
        if set(visited_nodes) == set(all_satellites):
            break

        # Print current step
        print(f"The cheapest next option from {start_satellite} is to {next_satellite} with a light travel time of {travel_time:.3f} minutes.")

        # Update the requested time, adding travel time
        requested_time = add_minutes_to_time(requested_time, travel_time)
        print(f"Arrival time at {next_satellite}: {requested_time}")

        # Add the satellite and corresponding time to the route
        visited_nodes.append(next_satellite)
        times.append(requested_time)

    # Print the final route summary
    print_route(visited_nodes, times)

if __name__ == '__main__':
    main()

