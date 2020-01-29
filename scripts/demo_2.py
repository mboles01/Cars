#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:30:31 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Cars/scripts') 

# open full listings data set and depreciation summary
import pandas as pd
listings_data = pd.read_csv('../data/listings4.csv')
depr_summary = pd.read_csv('../data/depreciation/depreciation_by_model.csv')





### PRE-PROCESS DATA ###

# exclude models older than 
newerthan = 1995
listings_data_filtered = listings_data[listings_data['Year'] > newerthan]

# filter fit data to exclude poor fit quality
depr_summary_filtered = depr_summary[depr_summary['Fit_age_R2'] > 0.67]

# create dropdown menu containing make/model sorted by frequency
model_counts = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)

# view all make/model combinations with minimum R2 fit quality - 828->156
model_counts_filtered = depr_summary_filtered.merge(model_counts.reset_index(), on='Model', how='left')






### CREATE COMBINATION PLOT ###

# random generator
selection = model_counts_filtered  # fully random
selection = model_counts_filtered[model_counts_filtered['Counts'] > 750] # many counts
import numpy.random as npr
model = selection.iloc[npr.randint(0,len(selection))][2]

# plot 
from plotfunctions_3 import plot_combo_depr2
plot_combo_depr2(listings_data_filtered, 
                 depr_summary_filtered, 
                 model, 
                 model_counts, 
                 save=False)

data = listings_data_filtered
depr_summary = depr_summary_filtered



