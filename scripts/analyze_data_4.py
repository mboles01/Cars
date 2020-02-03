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
ylabel = 'Counts'
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

fig, ax = plt.subplots(1, 1, figsize = (20,20))

ax = sns.boxplot(x = depr_order_brand.index, y = depr_order_brand['Half life'], data = depr_order_brand, 
                 showfliers = False, order = list(depr_order_brand.index), linewidth = 5)
ax = sns.stripplot(x = depr_order_brand.index, y = depr_order_brand['Half life'], data = depr_order_brand,
                 order = list(depr_order_brand.index), jitter = 0.25, size = 15,
                 linewidth = 3, edgecolor = 'black', alpha = 0.5)

# set axis properties
plt.xticks(rotation=45, fontname = 'Helvetica', fontsize = 42, ha = 'right')
plt.yticks(fontname = 'Helvetica', fontsize = 42)


plt.xlabel('Body type', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Half life (years)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(2, 8); ax.yaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3)

figure_name = '../images/depr_across_segment_R2>0.67.png'

plt.tight_layout()
plt.show()

plt.savefig(figure_name, dpi = 150)




