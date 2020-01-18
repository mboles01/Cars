#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:35:05 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Depreciator/scripts') 

# point to directory containing many raw data files
import os.path
import glob
mypath = r'../data/raw/New_2017-2020' # r'../data/raw/Used_1980-2020'

all_files = glob.glob(mypath + "/*.csv")

# load all csv files into one pandas dataframe (concatenating)
import pandas as pd
data_raw = pd.concat((pd.read_csv(f) for f in all_files))

# loop through each line in raw data
import ast
from helper_functions import flatten

data_df = pd.DataFrame()
for line in data_raw['0'].iloc[:-1]:
    
    # convert string to dictionary
    line_dict = ast.literal_eval(line)
    
    # flatten dictionary
    line_flat = flatten(line_dict)
    
    # convert to dictionary
    line_flat_df = pd.DataFrame(line_flat, index=[0])
    
    # append to existing dataframe
    data_df = data_df.append(line_flat_df, ignore_index=True)
    
# save dataframe as .csv
data_df.to_csv('../data/clean/new_listings_clean_1.csv')


# open clean_1 dataframe
import pandas as pd
listings_clean1 = pd.read_csv('../data/clean/new_listings_clean_1.csv')

# drop columns 
listings_clean1.columns
listings_clean2 = listings_clean1.drop(columns=['Unnamed: 0',
                                                '@context',
                                                '@type',
                                                'image',
                                                'mileageFromOdometer_@type',
                                                'offers_@type',
                                                'offers_availability',
                                                'offers_itemCondition',
                                                'offers_priceCurrency'])

# reorder columns

cols2 = ['productionDate', 
         'brand', 
         'model', 
         'mileageFromOdometer_value', 
         'offers_price', 
         'bodyType',
         'color', 
         'vehicleInteriorColor', 
         'driveWheelConfiguration', 
         'vehicleTransmission', 
         'vehicleEngine',  
         'name', 
         'url',
         'vehicleIdentificationNumber',
         'description']

listings_clean3 = listings_clean2[cols2]

# rename columns

listings_clean4 = listings_clean3.rename(columns={'productionDate': 'Year', 
                                'brand': 'Make',
                                'model': 'Model',
                                'mileageFromOdometer_value': 'Mileage',
                                'offers_price': 'Price',
                                'vehicleIdentificationNumber': 'VIN',
                                'color': 'Color',
                                'vehicleInteriorColor': 'InteriorColor',
                                'driveWheelConfiguration': 'Drive',
                                'vehicleTransmission': 'Transmission',
                                'vehicleEngine': 'Engine',
                                'bodyType': 'Body',
                                'name': 'ListTitle',
                                'url': 'URL',
                                'description': 'Description'})  


# save dataframe as .csv
listings_clean4.to_csv('../data/clean/listings_clean_4.csv')


# plot histogram
prices = listings_clean4['Price']

import numpy as np
prices_mean = round(np.average(prices))
prices_median = np.median(prices)
prices_stdev = np.std(prices)


from plotfunctions import plothist
binwidth = 1000

prices_textbox = 'Average = $%.0f \nMedian = $%.0f \nStdev = $%.0f' % (prices_mean, prices_median, prices_stdev)
plothist(prices, binwidth, prices_textbox, 0, 70000, 'Price ($)', 'Counts', 'Prices.png')






