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

### CREATE PLOTS FOR MVP ###

# create dropdown menu containing make/model sorted by frequency
# make_model = listings_data.groupby(['Make', 'Model']).size().to_frame().rename(columns={'0': 'Counts'})
model_counts = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)

# view all make/model combinations with minimum R2 fit quality - 828->156
model_counts_filtered = fit_data_filtered.merge(model_counts.reset_index(), on='Model', how='left')


### user input ###
data = listings_data
newerthan = 1995
model = 'Accord'

# random generator
import numpy.random as npr
model = model_counts_filtered.iloc[npr.randint(0,len(model_counts_filtered))][2]
print(model)



### first plot: price versus age with fit ###
counter = 1
from plotfunctions import plot_depr_age
plot_depr_age(data, model, newerthan, counter, save=False)




### second plot: depreciation in context of segment ###
# model = 'Accord' # input

# from plotfunctions_2 import plot_depr_age_segment
# plot_depr_age_segment(model, emp_data, fit_data_filtered, pred_data, save=False)

from plotfunctions_2 import plot_depr_age_segment2
plot_depr_age_segment2(model, fit_data_filtered, pred_data, save=False)



### third plot: table of depreciation of selected model versus others
# model = 'Accord' # input

# get depreciation rate of selected model
make = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Make'].iloc[0]
b = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Fit_age_b'].iloc[0]
halflife = 0.693/b

print('Fitting list prices to P(t) = a*exp(-bt)+c indicates that ')
print('the ' + str(make) + ' ' + str(model))
print('has b = ' + str(round(b,3)) + ' - it loses half its value every ' + str(round(halflife,1)) + ' years')


# get comparison data - top n cars (by counts) within segment
n = 10
vehicle_segment = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Body'].iloc[0]
segment_top_n = fit_data_filtered[fit_data_filtered['Body'] == str(vehicle_segment)][:n]

# pull out parameters of interest, add half life column
depr_top_n = segment_top_n[['Make', 'Model', 'Fit_age_b', 'Fit_age_R2']]
depr_top_n['Half life'] = 0.693/depr_top_n['Fit_age_b']
depr_top_n = depr_top_n.sort_values('Half life', ascending=False)


### CREATE COMBINATION PLOT ###

# random generator
selection = model_counts_filtered[:125]
import numpy.random as npr
model = selection.iloc[npr.randint(0,len(selection))][2]

from plotfunctions_3 import plot_combo_depr
plot_combo_depr(listings_data, fit_data_filtered, pred_data, model, newerthan, counter, save=False)







