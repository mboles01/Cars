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
fit_data = pd.read_csv('../data/depreciation/depreciation_all_models/fit_data_6_clean.csv')
emp_data = pd.read_csv('../data/depreciation/depreciation_all_models/emp_data_6.csv')
pred_data = pd.read_csv('../data/depreciation/depreciation_all_models/pred_data_6.csv')

# filter fit data to exclude poor fit quality
fit_data_filtered = fit_data[fit_data['Fit_age_R2'] > 0.67]
counter = 1
newerthan = 1995

### CREATE PLOTS FOR MVP ###

# create dropdown menu containing make/model sorted by frequency
model_counts = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)

# view all make/model combinations with minimum R2 fit quality - 828->156
model_counts_filtered = fit_data_filtered.merge(model_counts.reset_index(), on='Model', how='left')

### CREATE COMBINATION PLOT ###

# random generator
selection = model_counts_filtered[:100]
import numpy.random as npr
model = selection.iloc[npr.randint(0,len(selection))][2]

from plotfunctions_3 import plot_combo_depr
plot_combo_depr(listings_data, fit_data_filtered, pred_data, model, newerthan, counter, save=False)







