#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:30:31 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Depreciator/scripts') 

# open new and used listings "clean_1" dataframes
import pandas as pd
all_listings_clean2 = pd.read_csv('../data/clean/all_listings_clean_2.csv')



# ### PLOT DEPRECIATION CURVES ###

# # load plot function and listings data
# from plotfunctions import plot_depr_age
# data = all_listings_clean2

# # # collect top n models by count frequency
# model_counts = all_listings_clean2.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)
# # selection = model_counts[:5]

###

# create depreciation tables across models
from fit_functions import fit_depr

model_counts = all_listings_clean2.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)
selection = model_counts[:300]
data = all_listings_clean2
newerthan = 1995
bounds_age = ((10000, 0.05, 0), (200000, 1, 50000))
bounds_miles = ((10000, 0, 0), (200000, .003, 50000))

# remove problematic listings
selection = selection.drop(['Niro', 'e-Golf', 'M340i', 'Atlas', 'EcoSport', 'Supra', 'Clarity', 'Blazer', 'Mirage G4', 'Pickup'])

fit_data = pd.DataFrame()
emp_data = pd.DataFrame()
pred_data = pd.DataFrame()
for counter, line in enumerate(selection.index,1):
    print(counter, line)
    model = line
    fit_data, emp_data, pred_data = fit_depr(data, model, newerthan, counter, 
                                             fit_data, emp_data, pred_data, 
                                             bounds_age, bounds_miles)    

fit_data.to_csv('../data/depreciation/depreciation_all_models/fit_data_5.csv', index=False)
emp_data.sort_values(by=['Age']).to_csv('../data/depreciation/depreciation_all_models/emp_data_5.csv', index=False)
pred_data.to_csv('../data/depreciation/depreciation_all_models/pred_data_5.csv', index=False)






# plot age depreciation curves for selected models
cars = selection
fit_data_age = pd.DataFrame()
for counter, line in enumerate(cars.index,1):
    print(counter, line)
    model = line
    try:
        fit_data_age = plot_depr_age(data, model, newerthan, counter, fit_data_age)
    except Exception:
        print('Exception')
        continue


###

# plot miles depreciation curves for selected models
from plotfunctions import plot_depr_miles

cars = selection
fit_data_miles = pd.DataFrame()
for counter, line in enumerate(cars.index,1):
    print(counter, line)
    model = line    
    try:
        fit_data_miles = plot_depr_miles(data, model, newerthan, counter, fit_data_miles)
    except Exception:
        print('Exception')
        continue

