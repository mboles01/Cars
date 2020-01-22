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
fit_data = pd.read_csv('../data/depreciation/fit_data_3.csv')

### PLOT BAR CHART ###

# R2 vs. P(t), P(m), P(t)_cw for top 12 make/model combinations

