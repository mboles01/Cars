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

# ### PLOT SUMMARY DATA ###

# # pull prices for histogram
# prices = all_listings_clean2['Price']

# # set up statistics text box
# import numpy as np
# prices_mean = round(np.average(prices))
# prices_median = np.median(prices)
# prices_stdev = np.std(prices)
# prices_textbox = 'Average = $%.0f \nMedian = $%.0f \nStdev = $%.0f' % (prices_mean, prices_median, prices_stdev)

# # plot prices histogram
# from plotfunctions import plot_hist
# binwidth = 1000
# plot_hist(prices, 
#          binwidth, 
#          prices_textbox, 
#          0, 70000, 
#          'Price ($)', 
#          'Counts', 
#          '../images/Prices.png')


### PLOT DEPRECIATION CURVES ###

# load plot function and listings data
from plotfunctions import plot_depr_age
data = all_listings_clean2

# exclude models older than 
newerthan = 1995

# # select one make/model combination
# model = 'Civic'
# plot_depr(data, model, newerthan)

# collect top n models by count frequency
model_counts = all_listings_clean2.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)
selection = model_counts[:5]

# plot age depreciation curves for selected models
cars = selection
fit_data = pd.DataFrame()
for counter, line in enumerate(cars.index,1):
    print(counter, line)
    model = line
    try:
        fit_data_age = plot_depr_age(data, model, newerthan, counter, fit_data)
    except Exception:
        print('Exception')
        continue

# # rename fit data columns and save
# fit_data.columns=['Entry', 'Make', 'Model', 'a', 'b', 'c', 'R2 (median)', 'R2 (all)']
# fit_data.to_csv('../data/depreciation/fit_data_age.csv', index=False)

# plot miles depreciation curves for selected models
from plotfunctions import plot_depr_miles

cars = selection
fit_data_miles = pd.DataFrame()
for counter, line in enumerate(cars.index,1):
    print(counter, line)
    model = line
    fit_data_miles = plot_depr_miles(data, model, newerthan, counter, fit_data)
    
    try:
        fit_data_miles = plot_depr_miles(data, model, newerthan, counter, fit_data)
    except Exception:
        print('Exception')
        continue

# rename fit data columns and save
fit_data.columns=['Entry', 'Make', 'Model', 'a', 'b', 'c', 'R2 (median)', 'R2 (all)']
fit_data.to_csv('../data/depreciation/fit_data_miles.csv', index=False)












