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

import os
import numpy as np
import numpy.random as rand
import scipy as scipy
import random
import itertools
import pandas as pd
import skimage as si
import time
import my_functions as my_func

#TODO read in paths and number of threads using a gui interface
img_dir_path = my_func.convert_to_wsl_path(r"C:\Users\TracingPC1\Documents\zbrain_analysis\vglutGFP_individual_zbrain_stacks")
mask_dir_path = my_func.convert_to_wsl_path(r"C:\Users\TracingPC1\Documents\zbrain_analysis\MECE-Masks")
    
   
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
            mask_list.append([filename, mask])
            
    return mask_list

def get_masked_img_data(img_dir_path, mask_list):
    """Given a path to a directory containing the images to be checked, and a mask list, 
    decomposes the images into their masked means and stores them in a pandas dataframe."""
    
    masked_img_data_list = []
    
    # Iterate over all the images in the directory, skipping and subdirectories
    img_dir = os.fsencode(img_dir_path)
    for img_file in os.listdir(img_dir):
        filename = os.fsdecode(img_file)
        file_path = os.path.join(img_dir_path, filename)
        if os.path.isdir(file_path) == False:
            
            img = si.img_as_float(si.io.imread(file_path))
            dims = img.shape
            
            if len(dims)==3:
                img = img.reshape(dims[0],-1)
                
            elif len(dims)==2:
                continue
                
            else:
                print("Error: Invalid File Format")
                return None
                       
            temp_masked_img_data_list = []
            temp_masked_img_data_list.append(filename)
            for mask in mask_list:
                masked_img_mean = np.mean(img[mask[1].row,mask[1].col])
                temp_masked_img_data_list.append(masked_img_mean)
                
            masked_img_data_list.append(temp_masked_img_data_list)
            
    df = pd.DataFrame(masked_img_data_list)
    
    return df
                
# %% Testing reading in masks
    
start = time.time()
test_masks = read_masks(mask_dir_path)
elapsed = time.time()-start
print("Size of test_masks: " + str(getsizeof(test)))
print("time: " + str(elapsed))

# %% Testing masking images

start = time.time()
test_masked_img_means = get_masked_img_data(img_dir_path, test_masks)
elapsed = time.time()-start
print("Size of test_masked_img_means: " + str(getsizeof(test_masked_img_means)))
print("time: " + str(elapsed))


