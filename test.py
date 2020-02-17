#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:14:56 2020

@author: tracingpc1
"""
import my_functions as my_func
import skimage as si
import numpy as np
import scipy as scipy
from sys import getsizeof
import time

test_path = my_func.convert_to_wsl_path(r"C:\Users\TracingPC1\Documents\zbrain_analysis\MECE-Masks\CereBellum.tif")

test_image = si.io.imread(test_path).astype(bool)
start = time.time()
test_sparse = scipy.sparse.coo_matrix(test_image.reshape(138,-1))
elapsed = time.time()-start

print("test image: " + str(getsizeof(test_image)))
print("test sparse: " + str(getsizeof(test_sparse)))
print("time: " + str(elapsed))





#
#def get_ind_img_stats(dir_path, as_float):
#    #TODO run this live, debug and make sure it works as described
#    """Given a folder of images, calculates the mean and variance of each
#    individual image and returns a pandas data frame """
#    
#    img_dir = os.fsencode(dir_path)
#    img_data_list = [[]]
#    
#    for file in os.listdir(img_dir):
#        filename = os.fsdecode(file)
#        file_path = os.path.join(dir_path, filename)
#        
#        if as_float == True:
#            curr_img = si.img_as_float(si.io.imread(file_path))
#            img_mean = np.mean(curr_img)
#            img_var = np.var(curr_img)
#            img_data_list.append([filename, img_mean, img_var])
#            
#        elif as_float == False:
#            curr_img = si.io.imread(file_path)
#            img_mean = np.mean(curr_img)
#            img_var = np.var(curr_img)
#            img_data_list.append([filename, img_mean, img_var])
#            
#        data = pd.DataFrame(img_data_list[:][1:], rows=img_data_list[:][0], 
#                            columns=["Mean", "Variance"])
#        
#        #TODO create an accompanying funciton to plot this function
#        
#    return data

def get_ind_mask_stats(img_dir_path, mask_dir_path, as_float):
    """Given a folder of images and a folder of anatomical region masks, creates
    a pandas dataframe where each row is the means of each mask area for a given
    image, and the columns are the mean values of each image for a given mask
    region."""
    
    # Get a list of the image names
    img_dir = os.fsencode(img_dir_path)
    img_name_list = []
    for file in os.lsitdir(img_dir):
        img_name_list.append(os.fsdecode(file))
    
    # Get a list of the mask names
    mask_dir = os.fsencode(mask_dir_path)
    mask_name_list = []
    for file in os.listdir(mask_dir):
        mask_name_list.append(os.fsdecode(file))
        
    img_data_list = [[]]
    
    for img_file in os.listdir(img_dir):
        img_path = os.fsdecode(os.path.join(img_dir, img_file))
        
        if as_float = True:
            curr_img = si.img_as_float(si.io.imread(img_path))
            
        elif as_float = False:
            curr_img = si.io.imread(img_path)
            
        
    #TODO Iterate over each mask region for each file and save it as a nested list
    
    
    
    return None