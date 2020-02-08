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
mypath = r'../data/raw/' # r'../data/raw/Used_1980-2020'

all_files = glob.glob(mypath + "*/**/***.csv")

# load all csv files into one pandas dataframe (concatenating)
import pandas as pd
data_raw = pd.concat((pd.read_csv(f) for f in all_files))

# loop through each line in raw data
import ast
from helper_functions import flatten

# locale
import locale
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 






# clean all data and concatenate -- in 100 passes

passes = 100

start = 0
# i = 0
for i in range(0,passes,1):    
    stop = (i+1)*len(data_raw)//passes  
    print(str(i) + ' start: ' + str(start))
    print(str(i) + ' stop: ' +str(stop))   
    # for j in range(0,100,1):
        
    data_df = pd.DataFrame()
        
        # print('***' + str(i) + '***')
        
    for line in data_raw['0'][start:stop]:

            # print(round(len(data_df)/len(data_raw),4))
            # print(i)
            # print(j)
            
            # convert string to dictionary
            line_dict = ast.literal_eval(line)
            
            # flatten dictionary
            line_flat = flatten(line_dict)
            
            # convert to dictionary
            line_flat_df = pd.DataFrame(line_flat, index=[0])
            
            ### ADD COLUMNS ###
            
            # add condition (new/used) 
            line_flat_df['Condition'] = line_flat_df['name'].values[0].split()[0]
            
            # data_df = data_df.append(line_flat_df, ignore_index=True)

            
            if len(data_df)%25 == 0:
            # save dataframe as .csv
                data_df.to_csv('../data/clean/new/all_listings_clean_4' + '_' + str(i) + '.csv')
    
    # start = stop

            
            # add location (city, state)
            line_flat_df['Location'] = line_flat_df['description'].values[0].split('.')[0].split(': ')[1]
            
            # add options
            try:
                line_flat_df['Options'] = line_flat_df['description'].values[0].split('includes ')[1]
            except:
                line_flat_df['Options'] = ''
                
             
                
            # convert mileage string to int
            if line_flat_df['mileageFromOdometer_value'].iloc[0] != '':
                line_flat_df['mileageFromOdometer_value'] = locale.atoi(line_flat_df['mileageFromOdometer_value'].iloc[0])
            else: 
                line_flat_df['mileageFromOdometer_value'] == 0
                
            # append to existing dataframe
            data_df = data_df.append(line_flat_df, ignore_index=True)
        

            
            if len(data_df)%25 == 0:
                # save dataframe as .csv
                data_df.to_csv('../data/clean/new/all_listings_clean_4' + '_' + str(i) + '.csv')


    start = stop

