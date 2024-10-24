import pandas as pd
import numpy as np
from datetime import time

# SOLUTION 9:
def calculate_distance_matrix(df)->pd.DataFrame():
    # Extract unique IDs for the matrix
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))

    # Initialize the distance matrix with NaNs
    distance_matrix = pd.DataFrame(np.nan, index=ids, columns=ids)

    # Fill in the direct distances from the dataset
    for index, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    # Fill the diagonal with 0
    np.fill_diagonal(distance_matrix.values, 0)

    # Calculate cumulative distances for indirect routes
    for k in ids:
        for i in ids:
            for j in ids:
                if pd.notna(distance_matrix.at[i, k]) and pd.notna(distance_matrix.at[k, j]):
                    new_distance = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    if pd.isna(distance_matrix.at[i, j]) or new_distance < distance_matrix.at[i, j]:
                        distance_matrix.at[i, j] = new_distance

    return distance_matrix

# Example usage
# df2 = pd.read_csv('dataset-2.csv')
# distance_matrix = calculate_distance_matrix(df2)
# print(distance_matrix)


# SOLUTION  10:

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create a list to hold the unrolled data
    unrolled_data = []

    # Iterate over the index and columns of the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip if id_start and id_end are the same
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                # Only include valid distances (not NaN)
                if pd.notna(distance):
                    unrolled_data.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': distance
                    })

    # Convert the list of dictionaries to a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Example usage
# unrolled_df = unroll_distance_matrix(distance_matrix)
# print(unrolled_df)

#  SOLUTION 11:
def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter the DataFrame for the reference_id
    distances = df[df['id_start'] == reference_id]['distance']
    
    # Calculate the average distance for the reference_id
    if distances.empty:
        return []  # If there are no distances for the reference_id
    
    average_distance = distances.mean()

    # Calculate the 10% threshold
    lower_bound = average_distance * 0.90
    upper_bound = average_distance * 1.10

    # Find IDs within the threshold
    ids_within_threshold = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]
    
    # Get the unique id_start values and sort them
    result_ids = sorted(ids_within_threshold['id_start'].unique())

    return result_ids

# Example usage:
# reference_id = 1001400  # Example reference ID
# result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
# print(result_ids)

# SOLUTION 12:
def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define the rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type and add as new columns
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient
    
    return df

# Example usage:
# toll_rate_df = calculate_toll_rate(unrolled_df)
# print(toll_rate_df)

# SOLUTION 13:
def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define the time ranges and discount factors
    weekday_discount_factors = {
        'morning': 0.8, 
        'day': 1.2,     
        'evening': 0.8   
    }
    weekend_discount_factor = 0.7
    
    # Create new columns for day and time ranges
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create an empty list to hold the new rows
    new_rows = []

    # Iterate through each unique (id_start, id_end) pair in the DataFrame
    for _, group in df.groupby(['id_start', 'id_end']):
        for day in days_of_week:
            for hour in range(24):  # Loop through each hour
                start_time = time(hour, 0)  # Start at the beginning of each hour
                end_time = time(hour, 59)  # End at the end of each hour
                
                # Create a new row for each hour in the 24-hour period for the specific day
                new_row = {
                    'id_start': group['id_start'].iloc[0],
                    'id_end': group['id_end'].iloc[0],
                    'start_day': day,
                    'end_day': day,
                    'start_time': start_time,
                    'end_time': end_time,
                    'moto': group['moto'].iloc[0],
                    'car': group['car'].iloc[0],
                    'rv': group['rv'].iloc[0],
                    'bus': group['bus'].iloc[0],
                    'truck': group['truck'].iloc[0]
                }
                new_rows.append(new_row)

    # Creating new DataFrame from the list of new rows
    expanded_df = pd.DataFrame(new_rows)

    # Function to apply discounts based on day and time
    def apply_discount(row):
        if row['start_day'] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:  # Weekdays
            if row['start_time'] < time(10, 0):
                return row[['moto', 'car', 'rv', 'bus', 'truck']] * weekday_discount_factors['morning']
            elif time(10, 0) <= row['start_time'] < time(18, 0):
                return row[['moto', 'car', 'rv', 'bus', 'truck']] * weekday_discount_factors['day']
            else:
                return row[['moto', 'car', 'rv', 'bus', 'truck']] * weekday_discount_factors['evening']
        else:  # Weekends
            return row[['moto', 'car', 'rv', 'bus', 'truck']] * weekend_discount_factor

    # Apply discount based on day and time
    discounted_rates = expanded_df.apply(apply_discount, axis=1)
    
    # Update the toll rate columns with discounted values
    expanded_df[['moto', 'car', 'rv', 'bus', 'truck']] = discounted_rates
    
    return expanded_df

# Example usage:
# time_based_toll_rate_df = calculate_time_based_toll_rates(toll_rate_df)
# print(time_based_toll_rate_df)

