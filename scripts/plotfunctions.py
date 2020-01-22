#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:14:07 2019

@author: michaelboles
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt

# plot histogram

def plot_hist(data, binwidth, textbox, xmin, xmax, xlabel, ylabel, figure_name):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    bins = np.arange(round(min(data),1), max(data) + binwidth, binwidth)
    props = dict(facecolor='white', alpha=1.0)

    ax.hist(data, bins, edgecolor = 'black', facecolor = 'blue')
    
    plt.xlim(xmin, xmax); plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylabel(ylabel, fontsize = 18)
    ax.tick_params(axis = 'x', labelsize = 14); ax.tick_params(axis = 'y', labelsize = 14)
    ax.text(0.575, 0.97, textbox, transform = ax.transAxes, fontsize = 18, fontname = 'Helvetica', verticalalignment = 'top', bbox = props)
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.set_axisbelow(True)
    plt.savefig(figure_name, dpi = 600)
    plt.show()
    
    
# plot depreciation curve - age
    
def plot_depr_age(data, model, newerthan, counter, fit_data):
    
    import numpy as np
    import pandas as pd
    
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
    
    # # get fit quality data (diff bw median price and predicted value)
    # year_age_median_predicted_price = year_age_median_price.merge(price_predicted, on='Age', how='left')    
    # residuals_median = year_age_median_predicted_price['Predicted Price'] - year_age_median_predicted_price['Median Price']
    # ss_res_median = np.sum(residuals_median**2)   # residual sum of squares
    # ss_tot_median = np.sum((year_age_median_price['Median Price'] - np.mean(year_age_median_price['Median Price']))**2)   # total sum of squares
    # r_squared_median = 1 - (ss_res_median / ss_tot_median)
        
    # # store fit data in dataframe
    # fit_data_temp = pd.DataFrame(data = [[counter, make, model, popt[0], popt[1], popt[2], r_squared_median, r_squared_all]])
    # fit_data = pd.concat([fit_data, fit_data_temp], ignore_index=True)
        
    # plot scatter data
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.scatter(year_age_median_price['Age'], year_age_median_price['Median Price'])
    
    # set up figure axis
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.scatter(age_listprice['Age'], 
               age_listprice['List Price'], 
               edgecolor='black', facecolor='blue', alpha=0.33)
    
    # plot fit
    x_axis_smooth = np.arange(min(year_age_median_price['Age']), max(year_age_median_price['Age'])+1, .1)
    plt.plot(x_axis_smooth, exp_function(x_axis_smooth, *popt), '#ff4c00', linewidth=3)
    
    # set x- and y- labels
    xlabel = r'Age ($\it{t}$, years)'
    ylabel = r'Price ($\it{P}$, \$k)' 
    yscale = 1000

    plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylabel(ylabel, fontsize = 18, fontname = 'Helvetica')
    plt.title(str(make) + ' ' + str(model), fontsize = 20, fontname = 'Helvetica')
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    
    # force integers on x-axis
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # set up text box
    props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    
    textbox_1 = r'$P(t) = a{\bullet}exp(-bt) + c$'
    textbox_2 = '$a$ = %5.0f \n$b$ = %0.3f \n$c$ =%5.0f' % (popt[0], popt[1], popt[2]) + '\n$R^{2}$ = %5.2f' % r_squared_all
    
    ax.text(0.55, 0.95, textbox_1, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_1)
    
    ax.text(0.785, 0.825, textbox_2, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/yscale))
    ax.yaxis.set_major_formatter(ticks)
    
    # save figure, plot figure
    figure_name = '../images/depreciation/age_depreciation_plots_by_model/' + str(counter) + '_' + str(model) + '.png'
    plt.savefig(figure_name, dpi = 600)
    plt.show()

    # # return fit data
    # return(fit_data)



# plot depreciation curve - miles
    
def plot_depr_miles(data, model, newerthan, counter, fit_data):
        
    import pandas as pd
    import numpy as np

    # get model data from master list
    car_model_data_1 = data[data['Model'] == model]
    make = car_model_data_1['Make'].iloc[0]
    
    # filter data based on excluded years
    car_model_data_2 = car_model_data_1[car_model_data_1['Year'] > newerthan]
    
    # create dataframe with age and list price
    miles_listprice = pd.DataFrame({'Miles': car_model_data_2['Mileage'],
                              'List Price': car_model_data_2['Price'],
                              })   #.dropna()
    
        
    # fit data to function
    from scipy.optimize import curve_fit
    def exp_function(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    popt, pcov = curve_fit(exp_function, miles_listprice['Miles'], 
                           miles_listprice['List Price'], 
                           absolute_sigma=False, maxfev=1000,
                           bounds=((10000, 0, 0), (200000, .003, 50000)))
    
    # create predicted list price vs. miles 
    price_predicted = pd.DataFrame({'Miles': range(0,max(miles_listprice['Miles'].astype(int))+1,1),
                                    'Predicted Price': exp_function(range(0,max(miles_listprice['Miles'].astype(int))+1,1), 
                                                                    popt[0], popt[1], popt[2]), 
                                   })
    
    # combine list prices and predicted list prices
    miles_listprice_predprice = miles_listprice.merge(price_predicted, on='Miles', how='left')
    
    # calculate fit quality (diff bw all prices and predicted value)
    residuals_all = miles_listprice_predprice['List Price'] - miles_listprice_predprice['Predicted Price']
    ss_res_all = np.sum(residuals_all**2)   # residual sum of squares
    ss_tot_all = np.sum((miles_listprice_predprice['List Price'] - np.mean(miles_listprice_predprice['List Price']))**2)   # total sum of squares
    r_squared_all = 1 - (ss_res_all / ss_tot_all)
    
    # store fit data in dataframe
    fit_data_temp = pd.DataFrame(data = [[counter, make, model, popt[0], popt[1], popt[2], r_squared_all]])
    fit_data = pd.concat([fit_data, fit_data_temp], ignore_index=True)
        
    # plot scatter data
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.scatter(miles_listprice['Miles'], miles_listprice['List Price'])
    
    # set up figure axis
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.scatter(miles_listprice['Miles'], 
               miles_listprice['List Price'], 
               edgecolor='black', facecolor='blue', alpha=0.33)
    
    # plot fit
    x_axis_smooth = np.arange(min(miles_listprice['Miles']), max(miles_listprice['Miles'])+1, .1)
    plt.plot(x_axis_smooth, exp_function(x_axis_smooth, *popt), '#ff4c00', linewidth=3)
    
    # set x- and y- labels
    xlabel = r'Miles ($\it{m}$, k)'
    ylabel = r'Price ($\it{P}$, \$k)'
    xscale = 1000
    yscale = 1000

    plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylabel(ylabel, fontsize = 18, fontname = 'Helvetica')
    plt.title(str(make) + ' ' + str(model), fontsize = 20, fontname = 'Helvetica')
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    
    # # force integers on x-axis
    # from matplotlib.ticker import MaxNLocator
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # get b in scientific notation
    from decimal import Decimal
    b_sci = '%.3E' % Decimal(popt[1])
    
    # set up text box
    props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    
    textbox_1 = r'$P(m) = a{\bullet}exp(-bm) + c$'
    textbox_2 = '$a$ = %5.0f \n$b$ = %s \n$c$ =%5.0f' % (popt[0], float(b_sci), popt[2]) + '\n$R^{2}$ = %5.2f' % r_squared_all
    
    ax.text(0.5, 0.95, textbox_1, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_1)
    
    ax.text(0.72, 0.825, textbox_2, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
    # scale x- and y-axes
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/xscale))
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/yscale))
    ax.xaxis.set_major_formatter(ticks)
    ax.yaxis.set_major_formatter(ticks)
    
    # save figure, plot figure
    figure_name = '../images/depreciation/miles_depreciation_plots_by_model/' + str(counter) + '_' + str(model) + '.png'
    plt.savefig(figure_name, dpi = 600)
    plt.show()

    # # return fit data
    # return(fit_data)
