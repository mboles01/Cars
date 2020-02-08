#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:15:08 2020

@author: michaelboles
"""


# flatten nested dictionaries
import collections
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# get depreciation fit data
    
def fit_depr(data, model, newerthan, counter, fit_data):
    
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
    model_data_grouped_year = car_model_data_2.groupby('Year', as_index = False).median() # group prices within year by median value
    
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
                           bounds=((10000, 0.1, 0), (200000, 1, 100000)))
    
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
    
    # return fit data
    return(fit_data)
    
def exp_function(x, a, b, c):
    import numpy as np
    return a * np.exp(-b * x) + c



import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)

