#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:30:31 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Cars/scripts') 

# open listings dataframe
import pandas as pd
listings = pd.read_csv('../data/listings4.csv')
listings.columns



### ADDITIONAL CLEANING ###

# # drop duplicates
# len(listings)
# listings['VIN'].nunique()
# listings2 = listings.drop_duplicates(subset='VIN')
# listings2.to_csv('../data/listings2.csv', index=False)

# # get mileage column to numeric integer with zeros instead of NaN
# listings['Mileage'] = pd.to_numeric(listings['Mileage'], errors="coerce")
# listings['Mileage'] = listings['Mileage'].fillna(0)
# listings['Mileage'] = listings['Mileage'].astype(int)
# listings.to_csv('../data/listings3.csv', index=False)

# print make/model table
# listings['VIN'].nunique()

# # change 'Minivan' to 'Van'
# listings['Body'] = listings['Body'].replace({'Minivan':'Van'})
# listings.to_csv('../data/listings4.csv', index=False)







### CREATE MAKE/MODEL TABLE ###

# get unique columns
make_model_list = listings.groupby(['Make','Model', 'Body']).size().reset_index().rename(columns={0:'Count'})
make_model_list_sorted = make_model_list.sort_values(by = 'Count', ascending = False).replace('Sport Utility', 'SUV')
make_model_list_sorted.to_csv('../data/make_model_list_sorted.csv', index=False)

make_model_list_top = make_model_list_sorted[:500]

# sort to plot most popular models first

# # collect top n models by count frequency
listings_sorted = listings.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)

# check top listings
listings_sorted[:100]
listings.iloc[0]['Mileage'].astype(int)






### SCATTER PLOT: AGE VS. MILEAGE ###
listings_filtered = listings[listings['Year'] > 2004]
x_data = 2020 - listings_filtered['Year']
y_data = listings_filtered['Mileage']
x_lim = [0, 15]
y_lim = [0, 200000]
save = True

from plotfunctions import plot_age_miles
plot_age_miles(x_data, y_data, x_lim, y_lim, save)







### HISTOGRAM PLOT: PRICE ###
import numpy as np
listings_filtered = listings
data = listings_filtered['Price']
data_filtered = data[(data != np.inf) & (data != 0)]

# textbox
average = int(np.nanmean(data_filtered))
median = int(np.nanmedian(data_filtered))
stdev = int(np.std(data_filtered))   
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$List$ $price$ $(\$)$ \nAverage = %s \nMedian = %s \nStdev = %s' % (round(average,-2), 
                                                                     round(median,-2), 
                                                                     round(stdev,-2))

import numpy as np
binwidth = 1000
xmin = 0
xmax = 80000
ymin = 0
ymax = 4000
xlabel = 'Price ($k)'
ylabel = 'Counts'
figure_name = '../images/basic_eda/Price_hist_2.png'


import sys
sys.path.insert(0, "./data visualization/")
from plotfunctions_1 import plot_hist
plot_hist(data_filtered, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, figure_name)




### HISTOGRAM PLOT: MILES ###
import numpy as np
listings_filtered = listings
data = listings_filtered['Mileage']
data_filtered = data[data != np.inf]

# textbox
average = int(np.nanmean(data_filtered))
median = int(np.nanmedian(data_filtered))
stdev = int(np.std(data_filtered))   
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$Mileage$ $(miles)$ \nAverage = %s \nMedian = %s \nStdev = %s' % (round(average,-2), 
                                                                     round(median,-2), 
                                                                     round(stdev,-2))

import numpy as np
binwidth = 5000
xmin = 0
xmax = 250000
ymin = 0
ymax = 3500
xlabel = 'Miles (thousands)'
ylabel = 'Counts'
figure_name = '../images/basic_eda/Miles_hist_1.png'


import sys
sys.path.insert(0, "./data visualization/")
from plotfunctions_1 import plot_hist
plot_hist(data_filtered, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, figure_name)




### HISTOGRAM PLOT: MILES PER YEAR ###
import numpy as np  
data = listings_filtered['Mileage']/(2020 - listings_filtered['Year'])
data_filtered = data[(data != np.inf) & (data != 0)]


# textbox
average = int(np.nanmean(data_filtered))
median = int(np.nanmedian(data_filtered))
stdev = int(np.std(data_filtered))   
props = dict(facecolor='white', edgecolor='none', alpha=0.67)
textbox = '$Miles$ $per$ $year$ \nAverage = %s \nMedian = %s \nStdev = %s' % (round(average,-2), 
                                                                              round(median,-2), 
                                                                              round(stdev,-2))

# plot params
binwidth = 400
xmin = 0
xmax = 30000
ymin = 0
ymax = 3000
xlabel = 'Miles per year (k)'
ylabel = 'Counts'
figure_name = '../images/basic_eda/Miles_vs_age_hist_2.png'

from plotfunctions_1 import plot_hist
plot_hist(data_filtered, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, figure_name)







### BAR PLOT: MOST COMMON CARS ###
make_model_list_top = pd.read_csv('../data/make_model_list_sorted.csv')
common_cars = make_model_list_top[:20]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1,figsize=(10,7))
plt.bar(common_cars['Model'], common_cars['Count'], color='blue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.title('Most common cars', size=22)
plt.xlabel('Model', size=18)
plt.ylabel('Count', size=18)
# plt.tight_layout()
plt.savefig('../images/basic_eda/Common_cars_2.png', bbox_inches='tight', dpi = 200)








### BAR PLOT: MOST COMMON VEHICLE TYPE ###
body_list = listings.groupby(['Body']).size().reset_index().rename(columns={0:'Count'})
body_list = body_list.drop(body_list[body_list.Body == 'Unavailable '].index).sort_values('Count', ascending=False)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1,figsize=(7,7))
plt.bar(body_list['Body'], body_list['Count'], color='blue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.title('Body types', size=22)
plt.xlabel('Body type', size=18)
plt.ylabel('Count', size=18)
plt.tight_layout()
plt.savefig('../images/Body_type.png', dpi = 600)


