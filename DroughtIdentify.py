import numpy as np
import pandas as pd
import pyproj
import rasterio as rs
from shapely.geometry import Polygon
import os
from collections import deque
from functools import reduce
from pyproj import Geod


# Drought patch identification
def identify_drought(drought_index, threshold):
    '''
    BFS-based drought identification algorithm

    Input: 
    drought_index: Drought index matrix
    threshold: Drought threshold

    Output: 
    drought_label: Drought label matrix
    '''
    # Get the number of time steps, rows, and columns of the drought index matrix
    num, rows, cols = drought_index.shape
    
    # Create a drought label matrix with the same size as the drought index matrix, initialized to 0
    drought_label = np.zeros_like(drought_index)
    
    # Define a direction array to represent the surrounding 5 × 5 grid positions of each grid cell
    directions = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if i != 0 or j != 0]
    
    # Define a queue to store the labels of drought patches
    q = deque()
    
    # Initialize the drought patch label
    label_num = 1
    for t in range(num):
        # Traverse each grid cell in the drought index matrix
        for i in range(rows):
            for j in range(cols):
                # If the current grid cell has not been searched and the drought index is below the threshold, a drought has occurred
                if drought_label[t, i, j] == 0 and drought_index[t, i, j] < threshold:
                    # Add the drought patch label of the current grid cell to the queue and mark it as visited
                    q.append((t, i, j, label_num))
                    drought_label[t, i, j] = label_num
                
                    # Traverse each drought patch in the queue and search for surrounding drought grid cells
                    while q:
                        t, x, y, label = q.popleft()
                        # Traverse surrounding grid cells
                        for dx, dy in directions:
                            ii, jj = x + dx, y + dy
                            # If the grid cell has not been searched and the drought index is below the threshold, a drought has occurred
                            if ii >= 0 and ii < rows and jj >= 0 and jj < cols and drought_label[t, ii, jj] == 0 and drought_index[t, ii, jj] < threshold:
                                # Add the drought patch label of the current grid cell to the queue and mark it as visited
                                q.append((t, ii, jj, label))
                                drought_label[t, ii, jj] = label
                 

                    # When the queue is empty, a new drought patch has been fully identified, and the drought patch label is incremented
                    label_num += 1
            
    return drought_label       



# Determine temporal continuity of drought patches from the start
def drought_time_series(drought_labelled, min_area, t):    
    ''' 
    Input:
    drought_labelled: Initially identified drought patch label matrix
    min_area: Minimum detectable area
    t: Time length

    Output:
    drought_labelled: Temporally continuous drought label matrix
    '''
    # Traverse all time steps
    for t in range(1, t):
        # Current drought label matrix
        curr_drought_id = drought_labelled[t, :, :]
            
        # Previous drought label matrix
        prev_drought_id = drought_labelled[t-1, :, :]
            
        # Traverse all drought patches at the current time step
        for curr_id in np.unique(curr_drought_id[curr_drought_id > 0]):
            # Get the area of the current drought patch
            curr_area = np.sum(curr_drought_id == curr_id)
    
            # If the area of the current drought patch is smaller than the minimum area, it is not considered as a drought
            if curr_area < min_area:
                drought_labelled[drought_labelled == curr_id] = 0                
    
            # Traverse all drought patches at the previous time step
            for pre_id in np.unique(prev_drought_id[prev_drought_id > 0]):
                # Find the overlapping area of the two drought patches
                overlap_area = np.sum((curr_drought_id == curr_id) & (prev_drought_id == pre_id))
    
                # If the overlapping area is larger than the minimum drought area, they belong to the same drought event and merge the previous drought event
                if overlap_area >= min_area:
                    drought_labelled[drought_labelled == pre_id] = curr_id

    return drought_labelled


# Determine temporal continuity of drought patches from the end
def drought_time_series_reverse(drought_labelled, min_area, t):    
    ''' 
    Input:
    drought_labelled: Drought patch labels after temporal continuity determination from the start
    min_area: Minimum detectable area
    t: Time length

    Output:
    drought_labelled: Temporally continuous drought label matrix
    '''
    # Traverse all time steps
    for t in reversed(range(t-1)):
        # Current drought label matrix
        curr_drought_id = drought_labelled[t, :, :]
            
        # Next drought label matrix
        next_drought_id = drought_labelled[t+1, :, :]
            
        # Traverse all drought patches at the current time step
        for curr_id in np.unique(curr_drought_id[curr_drought_id > 0]):
            # Get the area of the current drought patch
            curr_area = np.sum(curr_drought_id == curr_id)
    
            # If the area of the current drought patch is smaller than the minimum area, it is not considered as a drought
            if curr_area < min_area:
                drought_labelled[drought_labelled == curr_id] = 0                
    
            # Traverse all drought patches at the next time step
            for next_id in np.unique(next_drought_id[next_drought_id > 0]):
                # Find the overlapping area of the two drought patches
                overlap_area = np.sum((curr_drought_id == curr_id) & (next_drought_id == next_id))
    
                # If the overlapping area is larger than the minimum drought area, they belong to the same drought event and merge the drought events
                if overlap_area >= min_area:
                    drought_labelled[drought_labelled == next_id] = curr_id

    return drought_labelled



# Get the drought duration
def drought_duration(drought_num_matrix):
    '''
    Input: 
    drought_num_matrix: Drought label matrix

    Output: 
    df: DataFrame — Drought number, start time, end time, duration
    '''
    droughts = []
    # Analyze each drought patch
    for num in np.unique(drought_num_matrix[drought_num_matrix > 0]):
        indices = list(zip(*np.where(drought_num_matrix == num)))
        start_t = indices[0][0]
        end_t = indices[-1][0]
        # Calculate the duration, add a random number in the range of -0.5 to 0.5
        # Formula (b-a)*np.random.random() + a
        duration = end_t - start_t + 1 + (0.5-(-0.5)) * np.random.random() + (-0.5)  
        droughts.append([num, start_t, end_t, duration])

    df = pd.DataFrame(droughts, columns=['DroughtNum', 'StartTime', 'EndTime', 'Duration'])
    return df



# Get the drought severity
def drought_severity(drought_num_matrix, drought_matrix, threshold):
    '''
    Input: 
    drought_num_matrix: Drought label matrix
    drought_matrix: Drought index matrix
    threshold: Drought threshold
    Output: 
    df: DataFrame — Drought number, drought severity
    '''

    droughts = []
    # Analyze each drought patch
    for num in np.unique(drought_num_matrix[drought_num_matrix > 0]):
        severity = 0
        indices = list(zip(*np.where(drought_num_matrix == num)))
        # Traverse each grid to calculate the drought severity
        for i in indices:
            severity += abs(drought_matrix[i] - threshold)
        droughts.append([num, severity])

    df = pd.DataFrame(droughts, columns=['DroughtNum', 'Severity'])
    return df



# Get the drought area
def drought_area(drought_num_matrix, dataset):
    '''
    Input: 
    drought_num_matrix: Drought label matrix
    dataset: The plane to get the coordinates

    Output: 
    df: DataFrame — Drought number, affected area

    '''

    geod = Geod(ellps="WGS84")

    droughts = []
    lat, lon = drought_num_matrix[0].shape
    # Analyze each drought patch
    for num in np.unique(drought_num_matrix[drought_num_matrix > 0]):
        # Generate a 2D projection plane
        drought_proj = np.zeros((lat, lon))
        # Initialize the drought patch area
        area = 0
        indices = list(zip(*np.where(drought_num_matrix == num)))
        for i in indices:
            drought_proj_site = i[1:3]
            # Check if the area of the current grid has been calculated
            if drought_proj[drought_proj_site] == 0:
                # Mark the grid cell
                drought_proj[drought_proj_site] = 1

                # Calculate the area of each grid cell
                
                # Get the coordinates of the four corners of the current grid cell (longitude, latitude)
                lon1, lat1 = dataset.transform * (i[2], i[1]+1)  # Top-right corner
                lon2, lat2 = dataset.transform * (i[2]+1, i[1]+1)  # Bottom-right corner
                lon3, lat3 = dataset.transform * (i[2]+1, i[1])  # Bottom-left corner
                lon4, lat4 = dataset.transform * (i[2], i[1])  # Top-left corner

                area0, zc = geod.geometry_area_perimeter(Polygon([
                    (lon1, lat1), (lon2, lat2),
                    (lon3, lat3), (lon4, lat4),
                ]))
            
                # Calculate the area of the polygon (in square meters)
                area += area0
                
        droughts.append([num, area])

    df = pd.DataFrame(droughts, columns=['DroughtNum', 'Area'])
    return df



if __name__ == "__main__":

    # Get file paths and all files in the directory

    parent_dir = "/data/ljp/result/" # the result directory
    path = "/data/ljp/SPI3/"  # the drought indices directory
    spi_list = os.listdir(path)
    
    # Read the first raster file to get the matrix size and data type
    with rs.open(path + spi_list[0]) as dataset:
        cols = dataset.width
        rows = dataset.height
        num = len(os.listdir(path))
        datatype = dataset.dtypes[0]

    # Create an empty 3D numpy array
    spi_data = np.zeros((num, rows, cols), dtype=datatype)
    
    # Read each raster file and store it in the 3D numpy array
    for i, spi_file in enumerate(spi_list):
    
        spi_filepath = path + spi_file
        # Open the raster file
        with rs.open(spi_filepath) as src:
            spi_band = src.read()
            spi_meta = src.meta
            spi_data[i, :, :] = spi_band

    drought_index = spi_data
    drought_index[drought_index < -50] = 1000

    # Define the drought index threshold
    threshold = -1
    # Get the drought patches
    drought_labels = identify_drought(drought_index, threshold)
    # Define the minimum area for drought recognition
    min_area = 12000
    # Determine temporal continuity
    drought_labelled_before = drought_time_series(drought_labels, min_area, num)
    # Determine temporal continuity — from end to start
    drought_labelled = drought_time_series_reverse(drought_labelled_before, min_area, num)
    
    # Save the drought label matrix for later use
    np.save(parent_dir + 'drought_labelled.npy', drought_labelled)

    print('Identification Done!!!')


    # Get the drought duration
    duration = drought_duration(drought_labelled)
    # Get the drought severity
    severity = drought_severity(drought_labelled, drought_index, threshold)
    # Get the drought area
    area = drought_area(drought_labelled, dataset)
    # Merge the drought features
    dfs = [duration, severity, area]    
    dfs_merged = reduce(lambda x, y: pd.merge(x, y, on="DroughtNum", how="outer"), dfs)

    # Filter droughts with a duration of more than 3 months
    df_selected = dfs_merged[dfs_merged['Duration'] > 3]

    # Re-sort by drought number
    drought_feature = df_selected.sort_values(by="DroughtNum")

    # Assign a new order (starting from 1)
    drought_feature.insert(0, 'Number', range(1, len(drought_feature)+1))

    # Save the results
    # Export to Excel  
    writer = pd.ExcelWriter(parent_dir + 'DroughtFeature.xlsx')

    drought_feature.to_excel(writer, index=False)

    writer.close()

    print("All Done!!!")




