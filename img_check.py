#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:23:22 2019

@author: Zachary Miller

This will be a self-contained tool that takes a folder of images and a folder
of anatomical region masks and uses both individual and total image statistics (RANSAC)
to find poor quality images, visualize these differences, and ultimately clean 
the image data of poor quality images
"""

# %% import packages, set file paths, define functions
import os
import numpy as np
import scipy as scipy
import itertools
import pandas as pd
import skimage as si
import time
import my_functions as my_func

#TODO read in paths and number of threads using a gui interface
img_dir_path = my_func.convert_to_wsl_path(r"C:\Users\TracingPC1\Documents\zbrain_analysis\vglutGFP_individual_zbrain_stacks")
mask_dir_path = my_func.convert_to_wsl_path(r"C:\Users\TracingPC1\Documents\zbrain_analysis\MECE-Masks")
n_threads = 8
outlier_threshold = 3

#%% 
   
def read_masks(mask_dir_path):
    """Given a path to a directory containing all of the mask files, reads in all of the masks
    (stored as sparse matrices, reshaped if originally 3d) along with their names into a list."""
    
    mask_list = []
    
    # Iterate over all the masks in the directory, skipping and subdirectories
    mask_dir = os.fsencode(mask_dir_path)
    for mask_file in os.listdir(mask_dir):
        filename = os.fsdecode(mask_file)
        file_path = os.path.join(mask_dir_path, filename)
        if os.path.isdir(file_path) == False:
            
            # Read in each mask as a boolean array 
            mask = si.io.imread(file_path).astype(bool)
            dims = mask.shape
            
            # Store the mask as a sparse matrix, reshaping first if necessary
            if len(dims) == 3:
                mask = scipy.sparse.coo_matrix(mask.reshape(dims[0],-1))
                
            elif len(dims) == 2:
                mask = scipy.sparse.coo_matrix(mask)
                
            else: 
                print("Error: Invalid File Format")
                return None
            
            #TODO trim the file extension off of filename
            name = os.path.splitext(filename)[0]
            mask_list.append([name, mask])
            
    return mask_list

def get_masked_img_data(img_dir_path, mask_list):
    """Given a path to a directory containing the images to be checked, and a mask list, 
    decomposes the images into their masked means and stores them in a pandas dataframe.
    Returns the dataframe"""
    
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
            #TODO try speeding this up by using JAX vmap
            for mask in mask_list:
                masked_img_mean = np.mean(img[mask[1].row,mask[1].col])
                temp_masked_img_data_list.append(masked_img_mean)
                
            masked_img_data_dict.update({name:temp_masked_img_data_list})
    
    # Create a dataframe and organize such that the row names are the image file names        
    df = pd.DataFrame.from_dict(masked_img_data_dict, orient="index")
    
    return df

def get_loo_vars(df):
    """Given a dataframe of the mask region means for n images, calculates the region
    variance for all possible combinations of n-1 images __returns outliers?, 
    loo stands for leave one out__"""
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
    

# %% Get DataFrame of mask region means
start = time.time()

print("Loading masks...")
#TODO add the column names in the funcion using columns= command upon creation
mask_list = read_masks(mask_dir_path)
df_header_list = [mask[0] for mask in mask_list]
print("Masks loaded.")
print("Calculating mask region means...")
mask_region_means_df = get_masked_img_data(img_dir_path, mask_list)
mask_region_means_df.columns = df_header_list
print("Region means calculated")

# %%
loo_ransac_region_var_df = get_loo_vars(mask_region_means_df)
loo_row_names_list = list(loo_ransac_region_var_df.index)

z_df = pd.DataFrame(scipy.stats.zscore(loo_ransac_region_var_df), 
                    index=loo_row_names_list, columns = df_header_list)
bad_regions_list = [(z_df.index[i], z_df.columns[j]) for i, j in 
                    np.argwhere(z_df.values<-1*outlier_threshold)]



elapsed = time.time()-start
print("time: " + str(elapsed))
                
## %% Testing reading in masks
#    
#start = time.time()
#test_masks = read_masks(mask_dir_path)
#elapsed = time.time()-start
#print("Size of test_masks: " + str(getsizeof(test)))
#print("time: " + str(elapsed))
#
## %% Testing masking images
#
#start = time.time()
#test_masked_img_means = get_masked_img_data(img_dir_path, test_masks)
#elapsed = time.time()-start
#print("Size of test_masked_img_means: " + str(getsizeof(test_masked_img_means)))
#print("time: " + str(elapsed))
#
## %%  Testing image variance calculations
#
#start = time.time()
#test = get_loo_vars(mask_region_means_df)
#elapsed = time.time()-start
##print("Size of test_masked_img_means: " + str(getsizeof(test_masked_img_means)))
#print("time: " + str(elapsed))
#
