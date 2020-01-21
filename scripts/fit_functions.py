#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:23:22 2020

@author: michaelboles
"""


lower_bounds = (10000, 0.1, 0)
upper_bounds = (200000, 1, 100000)


def fit_depr_age(data, model, newerthan, counter, fit_data):
        
    import pandas as pd
    import numpy as np
    
    # get model data from master list
    car_model_data_1 = data[data['Model'] == model]
    make = car_model_data_1['Make'].iloc[0]
    
    # filter data based on excluded years
    car_model_data_2 = car_model_data_1[car_model_data_1['Year'] > newerthan]
    
    # create dataframe with age and list price
    age_listprice = pd.DataFrame({'Age': 2020 - car_model_data_2['Year'],
                              'List Price': car_model_data_2['Price'],
                              })

    # group all listings by year, taking median value for curve fitting
    model_data_grouped_year = data.groupby('Year', as_index = False).median() # group prices within year by median value
    
    # create x- and y- columns for fit
    year_age_median_price = pd.DataFrame({'Year': model_data_grouped_year['Year'],
                               'Age': 2020 - model_data_grouped_year['Year'],
                               'Median Price': model_data_grouped_year['Price']
                               })
    
    # fit data to function
    from scipy.optimize import curve_fit
    def exp_function(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    popt, pcov = curve_fit(exp_function, year_age_median_price['Age'], 
                           year_age_median_price['Median Price'], 
                           absolute_sigma=False, maxfev=1000,
                           bounds=(lower_bounds, upper_bounds))
    
    # create predicted list price vs. age
    price_predicted = pd.DataFrame({'Age': range(0,max(year_age_median_price['Age'])+1,1),
                                    'Predicted Price': exp_function(range(0,max(year_age_median_price['Age'])+1,1), 
                                                                    popt[0], popt[1], popt[2]), 
                                   })
    
    # combine list prices and predicted list prices
    age_listprice_predprice = age_listprice.merge(price_predicted, on='Age', how='left')
    
    
    
    # calculate fit quality (diff bw all prices and predicted value)
    residuals_all = age_listprice_predprice['List Price'] - age_listprice_predprice['Predicted Price']
    ss_res_all = np.sum(residuals_all**2)   # residual sum of squares
    ss_tot_all = np.sum((age_listprice_predprice['List Price'] - np.mean(age_listprice_predprice['List Price']))**2)   # total sum of squares
    r_squared_all = 1 - (ss_res_all / ss_tot_all)
    
    # get fit quality data (diff bw median price and predicted value)
    year_age_median_predicted_price = year_age_median_price.merge(price_predicted, on='Age', how='left')    
    residuals_median = year_age_median_predicted_price['Predicted Price'] - year_age_median_predicted_price['Median Price']
    ss_res_median = np.sum(residuals_median**2)   # residual sum of squares
    ss_tot_median = np.sum((year_age_median_price['Median Price'] - np.mean(year_age_median_price['Median Price']))**2)   # total sum of squares
    r_squared_median = 1 - (ss_res_median / ss_tot_median)
        
    # store fit data in dataframe
    fit_data_temp = pd.DataFrame(data = [[counter, make, model, popt[0], popt[1], popt[2], r_squared_median, r_squared_all]])
    fit_data = pd.concat([fit_data, fit_data_temp], ignore_index=True)
