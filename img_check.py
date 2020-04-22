#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:23:22 2019

@author: Zachary Miller

This will be a self-contained tool that takes a folder of images and a folder
of anatomical region masks and uses both individual and total image statistics (RANSAC)
to find poor quality images, visualize these differences, and ultimately clean 
the image data of poor quality images

Need to convert to a module and then comment for Sumit. Probably just get it to
a point where he can customize it for use with his database and calculate all the
stats that he needs from it
"""

# %% import packages, set file paths, define functions
import os
import numpy as np
import scipy as scipy
from scipy import sparse
import itertools
import pandas as pd
import skimage as si
import time
import my_functions as my_func

#%% 

#TODO put the mask and image reading into one function with different type 
# options for the data input type... Note that this will require changing the
def getSparseMaskDictFromDir(mask_dir_path):
    """Loads all the masks in the given directory into sparse arrays
    and returns them as a dict along with their dimensions
    
    Args: 
        mask_dir_path (str): path to directory containing masks
        
    Returns:
        mask_dict (dict): dict containing mask name : sparse mask pairs
        original_dims (array): array of the mask dimensions formatted [z,y,x]
    """
    
    mask_dict = {}
    original_dims = 0
    
    # Iterate over all the masks in the directory, skipping any subdirectories
    mask_dir = os.fsencode(mask_dir_path)
    for mask_file in os.listdir(mask_dir):
        filename = os.fsdecode(mask_file)
        file_path = os.path.join(mask_dir_path, filename)
        mask_name = os.path.splitext(os.path.basename(file_path))[0]
        if os.path.isdir(file_path) == False:
            
            # Read in each mask as a boolean array 
            mask = si.io.imread(file_path).astype(bool)
            if original_dims==0:
                original_dims = mask.shape
            
            # Reshape the mask into a 2d array and save as a sparse array
            mask = sparse.coo_matrix(mask.reshape(mask.shape[0],-1))
            mask_dict.update({mask_name : mask})
    
    return mask_dict, original_dims


def getMaskedImgData(img_dir_path, mask_dict):
    """Decompose images in a directory into their mean values for each mask 
    region and return as a DataFrame
    
    Args:
        img_dir_path (str): path to directory containing images
        mask_dict (dict): dictionary of sparse masks formated {mask_name:mask}
        
    Returns:
        df_out (DataFrame): DataFrame with rows as images and columns as mask
                            mask means
        """
    
    masked_img_data_dict = {}
    
    # Iterate over all the images in the directory, skipping and subdirectories
    img_dir = os.fsencode(img_dir_path)
    for img_file in os.listdir(img_dir):
        filename = os.fsdecode(img_file)
        file_path = os.path.join(img_dir_path, filename)
        if os.path.isdir(file_path) == False:
            
            img = si.img_as_float(si.io.imread(file_path))
            dims = img.shape
            
            # Read the image into memory, reshaping if necessary
            if len(dims)==3:
                img = img.reshape(dims[0],-1)
                
            elif len(dims)==2:
                continue
                
            else:
                print("Error: Invalid File Format")
                return None
        
            # Get the name of the image and the mean of each mask region, append that to 
            # the main list
            temp_masked_img_data_list = []
            name = os.path.splitext(filename)[0]
            
            #TODO speed this up using paralellization and map
            for key in mask_dict.keys():
                mask = mask_dict[key]
                masked_img_mean = np.mean(img[mask.row,mask.col])
                temp_masked_img_data_list.append(masked_img_mean)
                
            masked_img_data_dict.update({name:temp_masked_img_data_list})
    
    # Create a dataframe and organize such that the row names are the image file names        
    masked_img_df = pd.DataFrame.from_dict(masked_img_data_dict, orient="index")
    
    return masked_img_df


def getLOOVars(df):
    """Calculates the leave-one-out varaince of each column of a data frame. 
    That is, the column variance of all n-1 combinations of the rows. Returns
    a DataFrame where the rows are the colun wise variances for that row row
    being left out of the calculation
    
    Args: 
        df (DataFrame): DataFrame for LOO variance calculation
        
    Returns:
        df_out (DataFrame): DataFrame with each row's LOO calculation
        """
    #TODO fix naming conventions
        
    # Get all possible combinations of n-1 row indices
    val_arr = df.values
    dims = val_arr.shape
    idx_arr = np.array(range(dims[0]))
    idx_arr_combs = np.asarray(list(itertools.combinations(idx_arr, dims[0]-1)))
    
    img_region_var_dict = {}
    for idx_vec in idx_arr_combs:
        # Get the missing image index and its name
        missing_idx = np.setdiff1d(idx_arr, idx_vec)
        img_name = df.index.values[missing_idx][0]
        
        # Calculate the varaince of all regions without this image
        val_arr_subset = np.delete(val_arr, missing_idx, axis=0)
        var_vals_list = list(np.var(val_arr_subset, axis=0))
        
        # Add the missing image name and the cooresponding region variances to the dict
        img_region_var_dict.update({img_name:var_vals_list})
        
    df_out = pd.DataFrame.from_dict(img_region_var_dict, orient="index", columns=df.columns)
     
    return df_out
    

#%% Set constants
##TODO test all this code with dummy data
    
#TODO fix naming conventions
img_dir_path = my_func.convert_to_wsl_path(r"C:\Users\TracingPC1\Documents\zbrain_analysis\vglutGFP_individual_zbrain_stacks")
mask_dir_path = my_func.convert_to_wsl_path(r"C:\Users\TracingPC1\Documents\zbrain_analysis\MECE-Masks")
OUTLIER_THRESHOLD = 3    

#%% Load masks and calculate mask means for each image
print("Loading masks...")

#TODO add the column names in the function
mask_dict = getSparseMaskDictFromDir(mask_dir_path)
df_header_list = [key for key in mask_dict.keys()]

print("Done\n")
print("Calculating mask region means...")

mask_region_means_df = getMaskedImgData(img_dir_path, mask_dict)
mask_region_means_df.columns = df_header_list

print("Done\n")

# %% Calculate the LOO variances
loo_ransac_region_var_df = getLOOVars(mask_region_means_df)
loo_row_names_list = list(loo_ransac_region_var_df.index)

# Calculate the z-scores of the region variances
z_df = pd.DataFrame(scipy.stats.zscore(loo_ransac_region_var_df, axis=1), 
                    index=loo_row_names_list, columns = df_header_list)
bad_regions_list = [(z_df.index[i], z_df.columns[j]) for i, j in 
                    np.argwhere(z_df.values<-1*OUTLIER_THRESHOLD)]
