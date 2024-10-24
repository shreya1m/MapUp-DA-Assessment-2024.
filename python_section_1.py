from typing import Dict, List , Any
import polyline 
import pandas as pd
import numpy as np
import re

# SOLUTION 1:
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    i = 0
    while i < len(lst):
        # Append elements in reverse order directly during the loop
        # Loop through the current group of size n or whatever is left of the list
        for j in range(min(i + n, len(lst)) - 1, i - 1, -1):
            result.append(lst[j])
        
        # Move to the next group
        i += n
    

    return result

# SOLUTION 2:
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {} 

    for string in lst:
        length = len(string)  

        # If the length is not in the dictionary, add it with an empty list
        if length not in length_dict:
            length_dict[length] = []
        
        # Append the string to the list corresponding to its length
        length_dict[length].append(string)
    
    # Sort the dictionary by key (length)
    return dict(sorted(length_dict.items()))

# SOLUTION 3:
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten(current: Any, parent_key: str = '') -> Dict:
        items = []
        if isinstance(current, dict):
            # If the current value is a dictionary, recursively flatten its contents
            for k, v in current.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(flatten(v, new_key).items())
        elif isinstance(current, list):
            # If the current value is a list, flatten each element using its index
            for i, v in enumerate(current):
                new_key = f"{parent_key}[{i}]"
                items.extend(flatten(v, new_key).items())
        else:
            # If it's neither a dict nor a list, it's a base case (e.g., a string or number)
            items.append((parent_key, current))
        return dict(items)
    
    return flatten(nested_dict)

# SOLUTION 4:
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])  # Making copy of the current path 
        
        for i in range(len(nums)):
            # Skip the current element if it's the same as the previous one and hasn't been used yet
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            if not used[i]:
                # Mark the current element as used
                used[i] = True
                path.append(nums[i])
                
                # Recurse with the updated path and used list
                backtrack(path, used)
                
                # (Backtrack) unmark the element and remove it from the path
                used[i] = False
                path.pop()

    nums.sort()  
    result = []
    used = [False] * len(nums)  # Used to track which elements have been used currently
    
    return result

# SOLUTION 5:
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define the regex pattern for the date formats
    pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    # Use re.findall to extract all matches of the pattern
    dates = re.findall(pattern, text)
    
    return dates

# SOLUTION 6:
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two latitude/longitude pairs."""
    R = 6371000  # Radius of the Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2)**2 + 
         np.cos(phi1) * np.cos(phi2) * 
         np.sin(delta_lambda / 2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Distance in meters

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)

    # Create a list for storing distances
    distances = [0]  # Distance for the first point is 0

    # Calculate distances using the Haversine formula
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i - 1]
        lat2, lon2 = coordinates[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)

    # Create a DataFrame from the coordinates and distances
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    df['distance'] = distances
    
    return df

# SOLUTION 7:
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Create a new matrix to hold the sums
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the row and column excluding the current element
            row_sum = sum(rotated_matrix[i])  # Sum of the ith row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the jth column
            
            # Exclude the Twice of current element as we added it twice
            final_matrix[i][j] = row_sum + col_sum - (2*rotated_matrix[i][j])
    
    return final_matrix
    

# SOLUTION 8:
def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Initialize an empty list to hold results
    results = []

    # Group by id and id_2
    for (id_val, id_2_val), group in df.groupby(['id', 'id_2']):
        # Get unique days from startDay and endDay
        start_days = set(group['startDay'].unique())
        end_days = set(group['endDay'].unique())
        
        # Check for all 7 days in startDay and endDay
        has_all_start_days = len(start_days) == 7
        has_all_end_days = len(end_days) == 7

        # Convert startTime and endTime to datetime
        group['start'] = pd.to_datetime(group['startTime'], format='%H:%M:%S')
        group['end'] = pd.to_datetime(group['endTime'], format='%H:%M:%S')

        # Calculate min and max time
        min_time = group['start'].min()
        max_time = group['end'].max()

        # Calculate time difference in hours
        time_difference = (max_time - min_time).total_seconds() / 3600.0

        # Check if the time difference covers at least 24 hours
        has_full_coverage = time_difference >= 24

        # Append results
        results.append(((id_val, id_2_val), has_all_start_days and has_all_end_days and has_full_coverage))

    # Create a boolean series with multi-index
    result_series = pd.Series(dict(results), name='Coverage Check')
    return result_series
# Example usage
# df = pd.read_csv('dataset-1.csv')
# result = time_check(df)
# print(result)
