import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io import load_file, save_file
import sys
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

class PointCloudDownsampler:
    def __init__(self, pc, vlength):
        self.pc = pc
        self.vlength = vlength
    
    def random_voxelisation(self):
        voxel_indices = np.floor(self.pc[:, :3] / self.vlength).astype(int)
        voxel_dict = defaultdict(list)
        for i, voxel_index in enumerate(voxel_indices):
            voxel_key = tuple(voxel_index)
            voxel_dict[voxel_key].append(i)  # Store the index instead of the point
        selected_indices = [voxel_points_indices[np.random.randint(len(voxel_points_indices))] for voxel_points_indices in voxel_dict.values()]
        return selected_indices

# Create an empty DataFrame to store the statistics
statistics_df = pd.DataFrame(columns=['File', 'Precision_fsct', 'Recall_fsct', 'Precision_ours', 'Recall_ours', 'Accuracy_fsct', 'Accuracy_ours', 'Weighted_accuracy_fsct', 'Weighted_accuracy_ours'])

# Get a list of all the file names in the folder
file_names = glob.glob(os.path.join(sys.argv[1], '*fsct.ply'))

# Wrap file_names with tqdm for the progress bar
for file_name in tqdm(file_names, desc='Processing files', unit='file'):
    # Load the ours and fsct point clouds
    base_name = os.path.basename(file_name).replace('_fsct.ply', '')
    ours_file = os.path.join(sys.argv[1], base_name + '_ours.ply')
    fsct_file = os.path.join(sys.argv[1], base_name + '_fsct.ply')

    ours = load_file(ours_file)
    ours.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x, inplace=True)
    ours = ours[(ours['label'] != 2)]
    fsct = load_file(fsct_file)
    fsct.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x, inplace=True)
    fsct = fsct[(fsct['label'] != 0) & (fsct['label'] != 2)]
    fsct['label'] = fsct['label'].replace({1: 0, 3: 1})

    # Assuming 'df' is your DataFrame
    if 'label' in ours.columns:
        # Get all occurrences of 'label' in the columns
        label_cols = [col for col in ours.columns if col == 'label']
        if len(label_cols) > 1:
            # Find the first occurrence and drop it
            first_label_index = ours.columns.get_loc('label')
            # Create a new DataFrame without the first 'label' column
            ours = pd.concat([
                ours.iloc[:, :first_label_index],  # Columns before the first 'label'
                ours.iloc[:, first_label_index+1:]  # Columns after the first 'label'
            ], axis=1)

        # Assuming 'df' is your DataFrame
    if 'label' in fsct.columns:
        # Get all occurrences of 'label' in the columns
        label_cols = [col for col in fsct.columns if col == 'label']
        if len(label_cols) > 1:
            # Find the first occurrence and drop it
            first_label_index = fsct.columns.get_loc('label')
            # Create a new DataFrame without the first 'label' column
            fsct = pd.concat([
                fsct.iloc[:, :first_label_index],  # Columns before the first 'label'
                fsct.iloc[:, first_label_index+1:]  # Columns after the first 'label'
            ], axis=1)
    
    # Downsample the point clouds
    # downsampler = PointCloudDownsampler(ours[['x','y','z']].values, 0.02)
    # ours = ours.iloc[downsampler.random_voxelisation()]
    # ours.reset_index(drop=True, inplace=True)
    
    # downsampler = PointCloudDownsampler(fsct[['x','y','z']].values, 0.02)
    # fsct = fsct.iloc[downsampler.random_voxelisation()]
    # fsct.reset_index(drop=True, inplace=True)
    
    # Compare the two point clouds and calculate the statistics
    f1_fsct = f1_score(fsct[['truth']].astype(int), fsct[['label']].astype(int), average='binary', zero_division=0)
    f1_ours = f1_score(ours[['truth']].astype(int), ours[['label']].astype(int), average='binary', zero_division=0)

    precision_fsct = precision_score(fsct[['truth']].astype(int), fsct[['label']].astype(int), average='binary', zero_division=0)
    recall_fsct = recall_score(fsct[['truth']].astype(int), fsct[['label']].astype(int), average='binary', zero_division=0)

    precision_ours = precision_score(ours[['truth']].astype(int), ours[['label']].astype(int), average='binary', zero_division=0)
    recall_ours = recall_score(ours[['truth']].astype(int), ours[['label']].astype(int), average='binary', zero_division=0)

    accuracy_fsct = balanced_accuracy_score(fsct[['truth']].astype(int), fsct[['label']].astype(int))
    accuracy_ours = balanced_accuracy_score(ours[['truth']].astype(int), ours[['label']].astype(int))

    print(file_name)
    print(f'Accuracy fsct: {accuracy_fsct}, Accuracy ours: {accuracy_ours}')

    if 'pathlength' not in fsct.columns:
        fsct['pathlength'] = 1
    if 'pathlength' not in ours.columns:
        ours['pathlength'] = 1

    weighted_accuracy_fsct = balanced_accuracy_score(fsct[['truth']].astype(int), fsct[['label']].astype(int), sample_weight=fsct['pathlength'])
    weighted_accuracy_ours = balanced_accuracy_score(ours[['truth']].astype(int), ours[['label']].astype(int), sample_weight=ours['pathlength'])

    # Create a DataFrame with the statistics
    statistics = pd.DataFrame({
        'File': [base_name],
        #'F1_fsct': [f1_fsct],
        #'F1_ours': [f1_ours],
        'Precision_fsct': [precision_fsct],
        'Recall_fsct': [recall_fsct],
        'Precision_ours': [precision_ours],
        'Recall_ours': [recall_ours],
        'Accuracy_fsct': [accuracy_fsct],
        'Accuracy_ours': [accuracy_ours],
        'Accuracy_weighted_fsct': [weighted_accuracy_fsct],
        'Accuracy_weighted_ours': [weighted_accuracy_ours]
    })

    # Concatenate the statistics DataFrame with the existing statistics_df DataFrame
    statistics_df = pd.concat([statistics_df, statistics], ignore_index=True)

import dataframe_image as dfi

# Define a function to replace country code in filename
def replace_country_code(filename):
    country_mapping = {'pol': 'Poland', 'spa': 'Spain', 'fin': 'Finland'}
    for code, country in country_mapping.items():
        if code in filename:
            return country
    return filename

# Create 'Country' column from 'File' column
statistics_df['Country'] = statistics_df['File'].str[:3].apply(replace_country_code)

# Drop 'File' column
statistics_df = statistics_df.drop(columns='File')

# Calculate the mean of numeric columns grouped by country
numeric_columns = statistics_df.select_dtypes(include=[np.number]).columns.tolist()
statistics_df = statistics_df.groupby('Country')[numeric_columns].mean().reset_index()

# Separate 'Country' column
country_df = statistics_df['Country']

# Drop 'Country' column from statistics_df
statistics_df = statistics_df.drop(columns='Country')

# Sort remaining columns by the first three characters
statistics_df = statistics_df.sort_index(axis=1, key=lambda x: x.str[:3])

# Concatenate 'Country' column back
country_df.reset_index(drop=True, inplace=True)
statistics_df.reset_index(drop=True, inplace=True)
statistics_df = pd.concat([country_df, statistics_df], axis=1).round(8)

# Replace underscore with space in column names
statistics_df.columns = statistics_df.columns.str.replace('_', ' ')

# Save the DataFrame to a CSV file
output_file = os.path.join(sys.argv[1], 'results.csv')
statistics_df.to_csv(output_file, index=False)

# Export the DataFrame as an image
dfi.export(statistics_df, os.path.join(sys.argv[1], 'results.png'))
