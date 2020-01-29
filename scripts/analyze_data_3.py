#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:30:31 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Depreciator/scripts') 

# open listings dataframe
import pandas as pd
listings = pd.read_csv('../data/listings4.csv')
listings.columns

# get sorted data
make_model_list_sorted = pd.read_csv('../data/make_model_list_sorted.csv')






# ## CREATE DEPRECIATION FITS ###

# # create depreciation tables across models
# from fit_functions_2 import fit_depr_2

# # # collect top n models by count frequency
# listings_sorted = listings.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)


# # pull out selection
# selection = listings_sorted[:300]
# newerthan = 1995
# bounds_age = ((10000, 0.01), (200000, 1))
# bounds_miles = ((10000, 0), (200000, .003))


# fit_data = pd.DataFrame()
# emp_data = pd.DataFrame()
# pred_data = pd.DataFrame()
# for counter, line in enumerate(selection.index,1):
#     model = line
#     print(model)
#     fit_data, emp_data, pred_data = fit_depr_2(listings, model, newerthan, counter, 
#                                                 fit_data, emp_data, pred_data, 
#                                                 bounds_age, bounds_miles)    

# fit_data.to_csv('../data/depreciation/fit_data.csv', index=False)
# emp_data.sort_values(by=['Age']).to_csv('../data/depreciation/emp_data.csv', index=False)
# pred_data.to_csv('../data/depreciation/pred_data.csv', index=False)






### PLOT R2 FOR 3 CHOICES OF DEPRECIATION MODELING ###

# import data, choose selection
fit_data = pd.read_csv('../data/depreciation/fit_data.csv')
selection = fit_data[:10]
selection.columns

# plot it
from plotfunctions import plot_depr_R2
plot_depr_R2(selection)






# ### CREATE DEPRECIATION SUMMARY ###

# # get depreciation fit data and filter by fit quality (R2)
# R2_min = 0
# depr_sorted = fit_data[['Make', 'Model', 'Fit_age_a', 'Fit_age_b', 'Fit_age_R2']].sort_values('Fit_age_b').reset_index()
# depr_sorted_filtered = depr_sorted[depr_sorted['Fit_age_R2'] > R2_min]

# # add body data to depreciation
# make_model = make_model_list_sorted[['Model','Body']]
# make_model_clean = make_model_list_sorted.drop_duplicates(subset='Model', keep='first')
# depr_with_body = depr_sorted_filtered.merge(make_model_clean[['Model','Body']], on='Model', how='left')

# # add half life to table, save .csv
# depr_with_body['Half life'] = 0.69315/depr_with_body['Fit_age_b']
# depr_with_body.sort_values('Make').to_csv('../data/depreciation/depreciation_by_model.csv', index=False)










### CREATE DEPRECIATION PLOTTER ###


# read in depreciation summary
depr_summary = pd.read_csv('../data/depreciation/depreciation_by_model.csv')

# user input: car choice
make_choices = depr_summary['Make'].unique()
import numpy as np
make_input = make_choices[np.random.randint(len(make_choices))]
# make_input = make_choices[4]

# choose model
model_choices = depr_summary[depr_summary['Make'] == make_input]['Model']
model_input = model_choices.iloc[np.random.randint(len(model_choices))]
# model_input = model_choices.iloc[2]

# get data from user choice
user_choice = depr_summary[depr_summary['Model'] == model_input]

# get segment of user choice
body = depr_summary[depr_summary['Model'] == model_input]['Body'].iloc[0]
segment = depr_summary[depr_summary['Body'] == body]









### HISTOGRAM PLOT: HALF LIFE ###
data_halflife = segment['Half life']

user_choice_halflife = user_choice['Half life'].iloc[0]

import numpy as np

# textbox
segment_average = np.nanmean(data_halflife)
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
line1 = '$Half$ $life$\n' + str(model_input) + ': %.2f\n' % user_choice['Half life'].iloc[0]
line2 = str(body) + ' average: %.2f' % segment_average                               
textbox = line1 + line2

import numpy as np
binwidth = .5
xmin = int(min(segment['Half life'])) - 1
xmax = int(max(segment['Half life'])) + 1
xlabel = 'Half life'
ylabel = 'Counts'
figure_name = '../images/Half_life_' + str(model_input) + '.png'

from plotfunctions import plot_hist_hl
bins = plot_hist_hl(data_halflife, user_choice_halflife, make_input, model_input, binwidth, textbox, props, xmin, xmax, xlabel, ylabel, figure_name)










### PLOT DEPRECIATION CURVES ###

listings_sorted = listings.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)
newerthan = 1995

# plot age depreciation curves for selected models
from plotfunctions import plot_depr_age
selection = listings_sorted[:1]
fit_data_age = pd.DataFrame()
for counter, line in enumerate(selection.index,1):
    print(counter, line)
    model = line
    counts = make_model_list_sorted[make_model_list_sorted['Model'] == model]['Count'].iloc[0]
    print(counts)
    fit_data_age = plot_depr_age(listings, model, newerthan, counter, counts, save=False)



###

# plot miles depreciation curves for selected models
from plotfunctions import plot_depr_miles

cars = selection
fit_data_miles = pd.DataFrame()
for counter, line in enumerate(cars.index,1):
    print(counter, line)
    model = line    
    fit_data_miles = plot_depr_miles(data, model, newerthan, counter, fit_data_miles)

    try:
        fit_data_miles = plot_depr_miles(data, model, newerthan, counter, fit_data_miles)
    except Exception:
        print('Exception')
        continue

