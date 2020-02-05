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
listings_data = pd.read_csv('../data/listings5.csv')
depr_summary = pd.read_csv('../data/depreciation/depreciation_by_model_2.csv')


### PRE-PROCESS DATA ###

# exclude models older than 
newerthan = 1995
listings_data_filtered = listings_data[listings_data['Year'] > newerthan]

# filter fit data to exclude poor fit quality
depr_summary_filtered = depr_summary[depr_summary['Fit_age_R2'] > 0.01]

# create dropdown menu containing make/model sorted by frequency
model_counts = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)

# view all make/model combinations with minimum R2 fit quality
model_counts_filtered = depr_summary_filtered.merge(model_counts.reset_index(), on='Model', how='left')
model_counts_filtered.to_csv('../data/model_counts_filtered.csv', index=False)


# check frequency of each body style
listings_data_filtered['Body'].value_counts()
depr_summary_filtered['Body'].value_counts()


# consolidate lower-frequency body styles into bigger bin
# listings_data_filtered.loc[(listings_data_filtered['Body'] == 'Convertible'),'Body'] = 'Coupe'
# listings_data_filtered.loc[(listings_data_filtered['Body'] == 'Hatchback'),'Body'] = 'Coupe'
# listings_data_filtered.loc[(listings_data_filtered['Body'] == 'Wagon'),'Body'] = 'Sedan'

depr_summary_filtered.loc[(depr_summary_filtered['Body'] == 'Convertible'),'Body'] = 'Coupe'
depr_summary_filtered.loc[(depr_summary_filtered['Body'] == 'Hatchback'),'Body'] = 'Coupe'
depr_summary_filtered.loc[(depr_summary_filtered['Body'] == 'Wagon'),'Body'] = 'Sedan'

# change to lowercase body style names
depr_summary_filtered.loc[(depr_summary_filtered['Body'] == 'Sedan'),'Body'] = 'sedan'
depr_summary_filtered.loc[(depr_summary_filtered['Body'] == 'Coupe'),'Body'] = 'coupe'
depr_summary_filtered.loc[(depr_summary_filtered['Body'] == 'Truck'),'Body'] = 'truck'
depr_summary_filtered.loc[(depr_summary_filtered['Body'] == 'Van'),'Body'] = 'van'


# change some things
# listings_data_filtered.loc[(listings_data_filtered['Model'] == 'Wrangler'),'Body'] = 'SUV'
depr_summary_filtered.loc[(depr_summary_filtered['Model'] == 'Wrangler'),'Body'] = 'SUV'
depr_summary_filtered.loc[(depr_summary_filtered['Model'] == 'Prius'),'Body'] = 'sedan'
depr_summary_filtered.loc[(depr_summary_filtered['Model'] == 'A7'),'Body'] = 'sedan'
depr_summary_filtered.loc[(depr_summary_filtered['Model'] == 'e-Golf'),'Body'] = 'sedan'
depr_summary_filtered.loc[(depr_summary_filtered['Model'] == 'Golf'),'Body'] = 'sedan'
depr_summary_filtered.loc[(depr_summary_filtered['Model'] == 'GTI'),'Body'] = 'sedan'
depr_summary_filtered.loc[(depr_summary_filtered['Model'] == 'Panamera'),'Body'] = 'sedan'
depr_summary_filtered.loc[(depr_summary_filtered['Model'] == 'C63 AMG'),'Body'] = 'coupe'

depr_summary_filtered = depr_summary_filtered.drop(depr_summary_filtered[depr_summary_filtered.Model == 'CT 200h'].index)
listings_data_filtered = listings_data_filtered.drop(listings_data_filtered[listings_data_filtered.Model == 'CT 200h'].index)
model_counts_filtered = model_counts_filtered.drop(model_counts_filtered[model_counts_filtered.Model == 'CT 200h'].index)
model_counts_filtered[105:110]

# # save .csv of filtered data
# listings_data_filtered.to_csv('../data/listings5_filtered.csv', index=False)
# depr_summary_filtered.to_csv('../data/depr_summary_filtered.csv', index=False)


### CREATE COMBINATION PLOT ###

# random generator
selection = model_counts_filtered  # fully random
# selection = model_counts_filtered[model_counts_filtered['Counts'] > 750] # many counts
# import numpy.random as np
# model = 'Accord'

# batch plot 
from plotfunctions_4 import plot_combo_depr2
for line in model_counts_filtered[105:].iterrows():
    
    model = line[1][2]
    print(model)
    
    plot_combo_depr2(listings_data_filtered, 
                         depr_summary_filtered, 
                         model, 
                         model_counts, 
                         save=False)

# single plot 
plot_combo_depr2(listings_data_filtered, 
                         depr_summary_filtered, 
                         model, 
                         model_counts, 
                         save=False)




# depreciation curve only
from plotfunctions import plot_depr_age
model = 'Volt'
newerthan = 1995
b_lower = 0.05
counter = 1
counts = model_counts[model_counts.index == str(model)].iloc[0][0]
alpha = 0.25
save = False
plot_depr_age(listings_data_filtered, 
              model, 
              newerthan,
              b_lower,
              counter, 
              counts,
              alpha,
              save)






