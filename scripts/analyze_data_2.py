#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:30:31 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Depreciator/scripts') 

# open listings data
import pandas as pd
listings_data = pd.read_csv('../data/clean/all_listings_clean_2.csv')
fit_data = pd.read_csv('../data/depreciation/depreciation_all_models/fit_data_4.csv')
emp_data = pd.read_csv('../data/depreciation/depreciation_all_models/emp_data_4.csv')

### CREATE PLOTS FOR MVP ###

# create dropdown menu containing make/model sorted by frequency
make_model = listings_data.groupby(['Make', 'Model']).size().to_frame().rename(columns={'0': 'Counts'})
model_counts = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)


### first plot: price versus age with fit ###

# input parameters of search
data = listings_data
newerthan = 1995
model = 'Civic'
counter = 1
fit_data = pd.DataFrame()

# plot data
from plotfunctions import plot_depr_age
plot_depr_age(data, model, newerthan, counter, fit_data, save=False)


### second plot: depreciation in context of others ###
