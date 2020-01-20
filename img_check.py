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
import random
import itertools
import pandas as pd
import skimage as si
import time


def get_ind_img_stats(dir_path, as_float):
    #TODO run this live, debug and make sure it works as described
    """Given a folder of images, calculates the mean and variance of each
    individual image and returns a pandas data frame """
    
    img_dir = os.fsencode(dir_path)
    img_data_list = [[]]
    
    for file in os.listdir(img_dir):
        filename = os.fsdecode(file)
        file_path = os.path.join(dir_path, filename)
        
        if as_float == True:
            curr_img = si.img_as_float(si.io.imread(file_path))
            img_mean = np.mean(curr_img)
            img_var = np.var(curr_img)
            img_data_list.append([filename, img_mean, img_var])
            
        elif as_float == False:
            curr_img = si.io.imread(file_path)
            img_mean = np.mean(curr_img)
            img_var = np.var(curr_img)
            img_data_list.append([filename, img_mean, img_var])
            
        data = pd.DataFrame(img_data_list[:][1:], rows=img_data_list[:][0], 
                            columns=["Mean", "Variance"])
        
        #TODO create an accompanying funciton to plot this function
        
    return data

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
