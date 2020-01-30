#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:36:20 2020

@author: michaelboles
"""


# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Cars/scripts') 

# open full listings data set and depreciation summary
import pandas as pd
listings = pd.read_csv('../data/listings3.csv')
depr_summary = pd.read_csv('../data/depreciation/depreciation_by_model_2.csv')


### PRE-PROCESS DATA ###

listings_short = listings_2[:25]
listings.columns

listings_2 = listings.drop(['ListTitle','URL', 'VIN'], axis=1)

location = listings_2['Description'].str.split('.', 1, expand=True)[0]
location = location.str.split(':', 1, expand=True)

listings_2['Location'] = location[1]

 

location = descriptions[0][0]



# exclude models older than 
newerthan = 1995
listings_data_filtered = listings_data[listings_data['Year'] > newerthan]

# filter fit data to exclude poor fit quality
depr_summary_filtered = depr_summary[depr_summary['Fit_age_R2'] > 0.67]

# create dropdown menu containing make/model sorted by frequency
model_counts = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)

# view all make/model combinations with minimum R2 fit quality
model_counts_filtered = depr_summary_filtered.merge(model_counts.reset_index(), on='Model', how='left')
