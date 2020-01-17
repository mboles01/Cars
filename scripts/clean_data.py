#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:35:05 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Depreciator/scripts') 

# point to directory containing many raw data files

import os.path
import glob
os.path.exists(r'../data/raw/Used_1980-2020')


mypath = r'../data/raw/Used_1980-2020'
all_files = glob.glob(mypath + "/*.csv")

# load all csv files into one pandas dataframe (concatenating)
import pandas as pd
data_raw = pd.concat((pd.read_csv(f) for f in all_files))


# loop through each line in raw data
import ast
from helper_functions import flatten

data_df = pd.DataFrame()
for line in data_raw['0'].iloc[:-1]:
    
    # convert string to dictionary
    line_dict = ast.literal_eval(line)
    
    # flatten dictionary
    line_flat = flatten(line_dict)
    
    # convert to dictionary
    line_flat_df = pd.DataFrame(line_flat, index=[0])
    
    # append to existing dataframe
    data_df = data_df.append(line_flat_df, ignore_index=True)
    
# save dataframe as .csv
data_df.to_csv('../data/clean/listings_clean_1.csv')





data_subset = data_raw[:100]
data_entry = data_subset.iloc[0][0]
type(data_entry)
a_dict = ast.literal_eval(data_entry)
type(a_dict)

line = data_raw['0'].iloc[:1][0]

data_subset.iloc[2][0]


a_flat = flatten(a)


a_df = pd.DataFrame.from_dict(a_flat)
