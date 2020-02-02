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
textbox = '$Half$ $life$ \nAverage = %.2f \nMedian = %.2f \nStdev = %.2f' % (average, median, stdev)

import numpy as np
binwidth = 0.5
xmin = 1
xmax = 11
ymin = 0
ymax = 35
xlabel = 'Half life'
ylabel = 'Counts'
figure_name = '../images/Depreciation_hist.png'

from plotfunctions import plot_hist
plot_hist(data, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, figure_name)
