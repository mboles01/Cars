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
listings_data = pd.read_csv('../data/listings5_filtered.csv')
depr_summary = pd.read_csv('../data/depr_summary_filtered.csv')


### HISTOGRAM PLOT: PRICE ###
import numpy as np
data = depr_summary['Half life']

# textbox
average = round(np.nanmean(data), 2)
median = round(np.nanmedian(data), 2)
stdev = round(np.std(data), 2)
props = dict(facecolor='white', edgecolor='none', alpha=0.67, boxstyle='square, pad=1')
textbox = '$Half$ $life$ $(years)$\nAverage = %.1f \nMedian = %.1f \nStdev = %.1f' % (average, median, stdev)

import numpy as np
binwidth = 0.5
xmin = 1
xmax = 11
ymin = 0
ymax = 35
xlabel = 'Half life (years)'
ylabel = 'Counts (number of car models)'
figure_name = '../images/Depreciation_hist.png'

from plotfunctions import plot_hist
plot_hist(data, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, figure_name)




### Box plots ###

# Box plot: depreciation across brands

# import data - depreciation across models
depr_summary = pd.read_csv('../data/depreciation/depreciation_by_model.csv')
depr_summary = pd.read_csv('../data/depr_summary_filtered.csv')

# remove some brands
indexNames = depr_summary[(depr_summary['Make'] == 'Maserati') | 
                          (depr_summary['Make'] == 'Jaguar') |
                          (depr_summary['Make'] == 'smart') |
                          (depr_summary['Make'] == 'Mitsubishi')].index

depr_summary.drop(indexNames, inplace=True)

# rename some brands
depr_summary.loc[(depr_summary['Make'] == 'MINI'),'Make'] = 'Mini'
depr_summary.loc[(depr_summary['Make'] == 'MAZDA'),'Make'] = 'Mazda'
depr_summary.loc[(depr_summary['Make'] == 'INFINITI'),'Make'] = 'Infinity'


# Determine order
depr_order_brand = depr_summary.groupby('Make').median().sort_values(by='Half life',ascending=True)

# create seaborn box + strip plot
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (45,22))

ax = sns.boxplot(x = 'Make', y = 'Half life', data = depr_summary, 
                 showfliers = False, order = list(depr_order_brand.index), linewidth = 5)
ax = sns.stripplot(x = 'Make', y = 'Half life', data = depr_summary,
                 order = list(depr_order_brand.index), jitter = 0.25, size = 15,
                 linewidth = 3, edgecolor = 'black', alpha = 0.5)

# set axis properties
plt.xticks(rotation=45, fontname = 'Helvetica', fontsize = 42, ha = 'right')
plt.yticks(fontname = 'Helvetica', fontsize = 42)


plt.xlabel('Car make', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Half life (years)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(2, 8); ax.yaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3)

figure_name = '../images/depr_across_brands_R2>0.67.png'

plt.tight_layout()
plt.savefig(figure_name, dpi = 150)




# Box plot: depreciation across segments

# Determine order
depr_order_brand = depr_summary.groupby('Body').median().sort_values(by='Half life',ascending=True)


# create seaborn box + strip plot
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (14,14))

ax = sns.boxplot(x = 'Body', 
                 y = 'Half life', 
                 data = depr_summary,
                 showfliers = False, 
                 order = list(depr_order_brand.index), 
                 linewidth = 5)

ax = sns.stripplot(x = 'Body', 
                   y = 'Half life', 
                   data = depr_summary,
                   order = list(depr_order_brand.index), 
                   jitter = 0.25, size = 15,
                   linewidth = 3, edgecolor = 'black', alpha = 0.5)

# set axis properties
import numpy as np
plt.xticks(np.arange(5), ('SUV', 'Sedan', 'Van', 'Coupe', 'Truck'),
    rotation=45, fontname = 'Helvetica', fontsize = 42, ha = 'right')
plt.yticks(fontname = 'Helvetica', fontsize = 42)
# plt.xticks(np.arange(5), ('SUV', 'Sedan', 'Van', 'Coupe', 'Truck'))

plt.xlabel('Body type', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Half life (years)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(0, 10); ax.yaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3)

figure_name = '../images/depr_across_segment_R2>0.67.png'

plt.tight_layout()

plt.savefig(figure_name, dpi = 150)

plt.show()




### FIT SURFACE ###



# # collect top n models by count frequency
listings_sorted = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)
camry = listings_data[listings_data['Model'] == 'Camry']

x = 2020 - camry['Year']
y = camry['Mileage']
z = camry['Price']


# create x- and y- columns for fit
price_year_age = pd.DataFrame({'Age': x,
                               'Mileage': y,
                               'Price': z,
                               })

# fit data to function
from scipy.optimize import curve_fit
def exp_function_2(x, y, a, b, c):
    return (a/2) * (np.exp(-b * x) + np.exp(-c * y))

popt, pcov = curve_fit(exp_function_2, 
                       price_year_age['Age'], 
                       price_year_age['Mileage'], 
                       price_year_age['Price'],
                       absolute_sigma=False, maxfev=1000)

                        # bounds=((10000, 0.1), (200000, 1)))



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x1, x2, y, alpha=0.25, edgecolor='black', color='blue')

ax.set_xlabel('Age')
ax.set_ylabel('Mileage')
ax.set_zlabel('Price')

# set axis limits
ax.axes.set_xlim3d(left=0, right=20) 
ax.axes.set_ylim3d(bottom=0, top=250000) 
ax.axes.set_zlim3d(bottom=0, top=40000) 

# rotate the axes and update
angle=30
ax.view_init(30, angle)

plt.savefig('../images/3d_plot_age_miles_price.png', dpi = 600)
plt.tight_layout()
plt.show()


# pull out selection
selection = listings_sorted[:10]


# 

