#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:17:13 2020

@author: tracingpc1
"""

def convert_to_wsl_path(path):
    # /mnt/c/Users/TracingPC1/Desktop/Bash_scripts/ants_apply_transformation.sh
    # "C:\Users\TracingPC1\Desktop\Bash_scripts\ants_apply_transformation.sh"
    new_path = path.strip("C:\\")
    new_path = new_path.replace("\\", "/")
    new_path = "".join(("/mnt/c/",new_path))
    return new_path