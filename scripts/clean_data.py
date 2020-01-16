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
for line in data_raw.iloc[0]:
    print(line)
    
data_raw.iloc[0][0]

# pull first 100 entries for testing
data_subset = data_raw[:100]
data_entry = data_subset.iloc[0][0]
type(data_subset.iloc[0][0])

# convert string to dictionary
import ast
a = ast.literal_eval(data_entry)
type(a)

# flatten nested dictionaries

import collections
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

a_flat = flatten(a)


a_df = pd.DataFrame.from_dict(a_flat)
