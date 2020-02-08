#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:23:22 2020

@author: michaelboles
"""


def fit_depr(data, model, newerthan, counter, 
             fit_data, emp_data, pred_data, 
             bounds_age, bounds_miles):
        
    import pandas as pd
    import numpy as np
    
    # get model data from master list
    car_model_data_1 = data[data['Model'] == model]
    make = car_model_data_1['Make'].iloc[0]
    body = car_model_data_1['Body'].iloc[0]
    
    # filter data based on excluded years
    car_model_data_2 = car_model_data_1[car_model_data_1['Year'] > newerthan]
    
    # create dataframe with age and list price
    age_listprice = pd.DataFrame({'Age': 2020 - car_model_data_2['Year'],
                              'List Price': car_model_data_2['Price'],
                              })
    
    # create dataframe with miles and list price
    miles_listprice = pd.DataFrame({'Miles': car_model_data_2['Mileage'],
                              'List Price': car_model_data_2['Price'],
                              })
    
    # group all listings by year, taking median value for curve fitting
    model_data_grouped_year = car_model_data_2.groupby('Year', as_index = False).median() # group prices within year by median value
    
    
    # create x- and y- columns for fit
    year_age_median_price = pd.DataFrame({'Year': model_data_grouped_year['Year'],
                               'Age': 2020 - model_data_grouped_year['Year'],
                               'Median Price': model_data_grouped_year['Price']
                               })
    
    # fit data to exponential function
    from scipy.optimize import curve_fit
    def exp_function(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    popt_age, pcov_age = curve_fit(exp_function, year_age_median_price['Age'], 
                           year_age_median_price['Median Price'], 
                           absolute_sigma=False, maxfev=1000,
                           bounds=(bounds_age[0], bounds_age[1]))
    
    popt_miles, pcov_miles = curve_fit(exp_function, miles_listprice['Miles'], 
                           miles_listprice['List Price'], 
                           absolute_sigma=False, maxfev=1000,
                           bounds=(bounds_miles[0], bounds_miles[1]))
    
    
    ### CREATE PRICE PREDICTIONS ###
    # try:
    # create predicted list price vs. age
    price_predicted_age = pd.DataFrame({'Age': range(0,max(year_age_median_price['Age'])+1,1),
                                    'Predicted Price': exp_function(range(0,max(year_age_median_price['Age'])+1,1), 
                                                                    popt_age[0], popt_age[1], popt_age[2])
                                    })
    
    # create predicted list price vs. mileage
    price_predicted_miles = pd.DataFrame({'Miles': range(0,max(car_model_data_2['Mileage'].astype(int))+1,1),
                                    'Predicted Price': exp_function(range(0,max(car_model_data_2['Mileage'].astype(int))+1,1), 
                                                                    popt_miles[0], popt_miles[1], popt_miles[2]) 
                                    })
    
# create predicted list price vs. age (conventional wisdom: 24% depr year 1, then 15% thereafter)
    if year_age_median_price[year_age_median_price['Age'] == 0].empty:
        r_squared_age_cw = np.nan
        
    else:
        new_price = year_age_median_price[year_age_median_price['Age'] == 0]['Median Price'].iloc[0]
        year_one_price = new_price*(1-0.24)
        year_n_price = year_one_price*(1-0.15)**(year_age_median_price['Age'][:-2]).iloc[::-1]
    
        price_predicted_cw = pd.Series([new_price, year_one_price]).append(year_n_price, ignore_index=True)    
        price_predicted_age_cw = pd.DataFrame({'Age': range(0,len(year_age_median_price['Age']),1),
                                'Predicted Price': price_predicted_cw
                                })
        
        age_listprice_predprice_cw = age_listprice.merge(price_predicted_age_cw, on='Age', how='left')
        
        # calculate fit quality for price vs. age (diff bw all prices and predicted value)
        residuals_age_cw = age_listprice_predprice_cw['List Price'] - age_listprice_predprice_cw['Predicted Price']
        ss_res_age_cw = np.sum(residuals_age_cw**2)   # residual sum of squares
        ss_tot_age_cw = np.sum((age_listprice_predprice_cw['List Price'] - np.mean(age_listprice_predprice_cw['List Price']))**2)   # total sum of squares
        r_squared_age_cw = 1 - (ss_res_age_cw / ss_tot_age_cw)
        
    
    # combine list prices and predicted list prices for age, mileage
    age_listprice_predprice = age_listprice.merge(price_predicted_age, on='Age', how='left')
    miles_listprice_predprice = miles_listprice.merge(price_predicted_miles, on='Miles', how='left')
    
    # calculate fit quality for price vs. age (diff bw all prices and predicted value)
    residuals_age = age_listprice_predprice['List Price'] - age_listprice_predprice['Predicted Price']
    ss_res_age = np.sum(residuals_age**2)   # residual sum of squares
    ss_tot_age = np.sum((age_listprice_predprice['List Price'] - np.mean(age_listprice_predprice['List Price']))**2)   # total sum of squares
    r_squared_age = 1 - (ss_res_age / ss_tot_age)
    
    # calculate fit quality for price vs. miles (diff bw all prices and predicted value)
    residuals_miles = miles_listprice_predprice['List Price'] - miles_listprice_predprice['Predicted Price']
    ss_res_miles = np.sum(residuals_miles**2)   # residual sum of squares
    ss_tot_miles = np.sum((miles_listprice_predprice['List Price'] - np.mean(miles_listprice_predprice['List Price']))**2)   # total sum of squares
    r_squared_miles = 1 - (ss_res_miles / ss_tot_miles)

    # create median price table for joining with other models
    median_price_temp = year_age_median_price[['Age', 'Median Price']].rename(columns={'Median Price': str(model)})
    
    # create price prediction table for joining with other models
    pred_price_temp = price_predicted_age.rename(columns={'Predicted Price': str(model)})
            
    # store fit data in dataframe
    fit_data_temp = pd.DataFrame({'Entry': counter,
                                  'Make': make,
                                  'Model': model,
                                  'Body': body,
                                  'Fit_age_a': popt_age[0], 
                                  'Fit_age_b': popt_age[1], 
                                  'Fit_age_c': popt_age[2],
                                  'Fit_age_R2': r_squared_age, 
                                  'Fit_miles_a': popt_miles[0], 
                                  'Fit_miles_b': popt_miles[1], 
                                  'Fit_miles_c': popt_miles[2],
                                  'Fit_miles_R2': [r_squared_miles],
                                  'Fit_age_cw_R2': r_squared_age_cw, 
                                  })
    
    
    # combine model-specific fit data with growing dataframes for each
    fit_data = pd.concat([fit_data, fit_data_temp], ignore_index=True)
    
    # combine model-specific median price data with growing dataframes for each
    if emp_data.empty:
        emp_data = median_price_temp
    else:
        emp_data = emp_data.merge(median_price_temp, on='Age', how='outer')
    
    
    # combine model-specific median price data with growing dataframes for each
    if pred_data.empty:
        pred_data = pred_price_temp
    else:
        pred_data = pred_data.merge(pred_price_temp, on='Age', how='outer')
        
    
    # return fit data and median price
    return(fit_data, emp_data, pred_data)                                  
                                  
    
    
    
    # except:
    #     print('*** Fit did not converge ***')
    #     pass
    
