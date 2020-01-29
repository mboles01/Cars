#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:13:43 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Depreciator/scripts') 

# point to directory containing many raw data files
import os.path
import glob
mypath = r'../data/clean/new/' 

all_files = glob.glob(mypath + "/*.csv")

# load all csv files into one pandas dataframe (concatenating)
import pandas as pd
data_raw = pd.concat((pd.read_csv(f) for f in all_files))

# save dataframe as .csv
data_raw.to_csv('../data/clean/new/all_listings_clean.csv', index=False)


# open new and used listings "clean_1" dataframes
import pandas as pd
all_listings_clean1 = pd.read_csv('../data/clean/new/all_listings_clean.csv')
all_listings_clean1['Condition'].unique()


# drop columns 
all_listings_clean1.columns
all_listings_clean2 = all_listings_clean1.drop(columns=['Unnamed: 0',
                                                '@context',
                                                '@type',
                                                'image',
                                                'mileageFromOdometer_@type',
                                                'offers_@type',
                                                'offers_availability',
                                                'offers_itemCondition',
                                                'offers_priceCurrency'])

all_listings_clean1['vehicleIdentificationNumber'].nunique()

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

all_listings_clean3 = all_listings_clean2[cols2]

# rename columns

all_listings_clean4 = all_listings_clean3.rename(columns={'productionDate': 'Year', 
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

# mileage column: replace all NaN with 0
# all_listings_clean4['Mileage'] = pd.to_numeric(all_listings_clean4['Mileage'].str.replace(',',''), errors='coerce').fillna(0)
all_listings_clean4['Mileage'] = all_listings_clean4['Mileage'].fillna(0)

# filter out entries with Price = 0 or Year = 0
all_listings_clean5 = all_listings_clean4[all_listings_clean4['Price'] != 0]
all_listings_clean6 = all_listings_clean5[all_listings_clean5['Year'] != 0]

# remove duplicates
all_listings_clean7 = all_listings_clean6.drop_duplicates().reset_index().drop(columns='index')

# save dataframe as .csv
all_listings_clean7.to_csv('../data/listings.csv', index=False)
