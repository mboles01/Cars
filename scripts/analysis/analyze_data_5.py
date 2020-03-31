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

# get sorted data
make_model_list_sorted = pd.read_csv('../data/make_model_list_sorted.csv')

# add location column
location = listings_data['Description'].str.split('.', 1, expand=True)[0]
location_2 = location.str.split(': ', 1, expand=True)
listings_data['Location'] = location_2[1]
listings_data.columns

# create fit
models = make_model_list_sorted[:10]['Model']

depr_by_location = pd.DataFrame()
for model in models:

    # model = 'Camry'
    model_data = listings_data[(listings_data['Model'] == str(model)) & (listings_data['Year'] > 1995)]
    
    cities = ['Chicago, IL', 'Los Angeles, CA', 'San Francisco, CA', 'New York, NY', 'Houston, TX']
    
    for city in cities:   
        
        selection = model_data[model_data['Location'] == city]
        
        # create dataframe with age and list price
        age_listprice = pd.DataFrame({'Age': 2020 - selection['Year'],
                                  'List Price': selection['Price'],
                                  })
        
        # group all listings by year, taking median value for curve fitting
        model_data_grouped_year = selection.groupby('Year', as_index = False).mean() # group prices within year by median value
        
        
        # create x- and y- columns for fit
        year_age_median_price = pd.DataFrame({'Year': model_data_grouped_year['Year'],
                                    'Age': 2020 - model_data_grouped_year['Year'],
                                    'Median Price': model_data_grouped_year['Price']
                                    })
    
        
        # fit data to function
        import numpy as np
        from scipy.optimize import curve_fit
        def exp_function(x, a, b):
            return a * np.exp(-b * x)
        
        popt, pcov = curve_fit(exp_function, year_age_median_price['Age'], 
                                year_age_median_price['Median Price'], 
                                absolute_sigma=False, maxfev=1000,
                                bounds=((10000, 0.05), (200000, 1)))
        
        
        half_life = round(0.6931/popt[1], 3)
        
        row_temp = pd.DataFrame({'Location': city,
                                'Model': model,
                                'Half life': [half_life]})
        
        depr_by_location = depr_by_location.append(row_temp, ignore_index=True)

depr_by_location.to_csv('depr_by_location.csv', index=False)

# plot depreciation by city across top ten models
        
import matplotlib.pyplot as plt

# set up plot
fig, ax = plt.subplots(1, 1, figsize=(12,10))
# plt.xlabel('Model', fontsize = 22, fontname = 'Helvetica')
plt.ylabel('Half-life (years)', fontsize = 24, fontname = 'Helvetica')
plt.ylim(0,11)

# set width of bar
barWidth = 0.125
 
# set height of bar
bars_ny = depr_by_location[depr_by_location['Location'] == 'New York, NY']
bars_la = depr_by_location[depr_by_location['Location'] == 'Los Angeles, CA']
bars_chi = depr_by_location[depr_by_location['Location'] == 'Chicago, IL']
bars_sf = depr_by_location[depr_by_location['Location'] == 'San Francisco, CA']
bars_h = depr_by_location[depr_by_location['Location'] == 'Houston, TX']

# Set position of bar on x-axis
r1 = np.arange(len(bars_ny)) - barWidth
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
 
# Make the plot
plt.bar(r1, bars_ny['Half life'], color='blue', width=barWidth, edgecolor='black', label='New York', zorder = 2)
plt.bar(r2, bars_la['Half life'], color='orange', width=barWidth, edgecolor='black', label='Los Angeles')
plt.bar(r3, bars_chi['Half life'], color='purple', width=barWidth, edgecolor='black', label='Chicago')
plt.bar(r4, bars_sf['Half life'], color='red', width=barWidth, edgecolor='black', label='San Francisco')
plt.bar(r5, bars_h['Half life'], color='green', width=barWidth, edgecolor='black', label='Houston')
 
# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(bars_ny))], 
           depr_by_location['Model'].unique().tolist(), 
           rotation=45, fontsize=22, 
           ha='right', fontname = 'Helvetica')
plt.yticks(fontname='Helvetica', fontsize=22)

# adjust tick label size
ax.tick_params(axis = 'x', labelsize = 20)
ax.tick_params(axis = 'y', labelsize = 20)
 
ax.yaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 2, length = 10)
ax.yaxis.set_tick_params(width = 2, length = 10)
plt.setp(ax.spines.values(), linewidth = 2)

# Create legend & Show graphic
plt.legend(prop={'size':22})
plt.title('Half-life by location', fontsize = 28, fontname = 'Helvetica')
plt.tight_layout()
plt.savefig('../images/half_life_by_location_cars', dpi = 150)
plt.show()



# Box plot: half life across cities

# Determine order
order = depr_by_location.groupby('Location').median().sort_values(by='Half life',ascending=True)

# create seaborn box + strip plot
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (14,14))

ax = sns.boxplot(x = 'Location', 
                 y = 'Half life', 
                 data = depr_by_location,
                 showfliers = False, 
                  order = list(order.index), 
                 linewidth = 5)

ax = sns.stripplot(x = 'Location', 
                   y = 'Half life', 
                   data = depr_by_location,
                    order = list(order.index), 
                   jitter = 0.125, size = 15,
                   linewidth = 3, edgecolor = 'black', alpha = 0.5)

# set axis properties
plt.xticks(rotation=45, fontname = 'Helvetica', fontsize = 42, ha = 'right')
plt.yticks(fontname = 'Helvetica', fontsize = 42)
# plt.xticks(np.arange(5), ('SUV', 'Sedan', 'Van', 'Coupe', 'Truck'))

plt.xlabel('Location', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Half-life (years)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(4, 10); ax.yaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3)

figure_name = '../images/half_life_by_location_cities.png'

plt.tight_layout()
plt.savefig(figure_name, dpi = 150)
plt.show()



