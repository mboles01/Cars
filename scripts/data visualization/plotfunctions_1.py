#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:14:07 2019

@author: michaelboles
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plot histogram

def plot_hist(data, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, figure_name):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    bins = np.arange(round(min(data),1), max(data) + binwidth, binwidth)
    props = dict(facecolor='white', alpha=1.0)

    ax.hist(data, bins, edgecolor = 'black', facecolor = 'blue')
    
    plt.xlim(xmin, xmax); plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel, fontsize = 18)
    ax.tick_params(axis = 'x', labelsize = 14); ax.tick_params(axis = 'y', labelsize = 14)
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
        
    ax.text(0.6, 0.95, textbox, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props)

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)

    fig.tight_layout()
    
    import matplotlib.ticker as ticker
    xscale = 1000
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/xscale))
    ax.xaxis.set_major_formatter(ticks)

    plt.savefig(figure_name, dpi = 600)
    plt.show()
    
    
    
    
    
    
    
def plot_age_miles(x_data, y_data, x_lim, y_lim, save):
    
    age_miles = pd.DataFrame({'Age': x_data,
                          'Miles': y_data})

    import matplotlib.pyplot as plt
    # set up figure axis
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.scatter(x_data, 
               y_data, 
               edgecolor='black', 
               facecolor='blue', 
               alpha=0.002)
    
    # set x- and y- labels
    xlabel = r'Age (years)'
    ylabel = r'Miles (thousands)' 
    yscale = 1000
    
    plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylabel(ylabel, fontsize = 18, fontname = 'Helvetica')
    # plt.title('Title', fontsize = 20, fontname = 'Helvetica')
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
    # adjust ticks
    import matplotlib.ticker as ticker
    import numpy as np
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/yscale))
    ax.yaxis.set_major_formatter(ticks)
    plt.xticks(np.arange(0, 16, step=3))
    plt.yticks(np.arange(0, 200001, step=50000))

    
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    
    # fit data to function
    from scipy.optimize import curve_fit
    def lin_function(x, b):
        return b*x
    
    popt, pcov = curve_fit(lin_function, 
                           x_data, 
                           y_data, 
                           absolute_sigma=False, maxfev=1000)
    
    # get R2 from fit
    miles_predicted = pd.DataFrame({'Age': range(0,max(x_data)+1,1),
                                    'Predicted Miles': lin_function(range(0,max(x_data)+1,1), 
                                                                    popt[0])})
    
    # combine real miles and predicted miles
    age_miles_predmiles = age_miles.merge(miles_predicted, on='Age', how='left')
    
    # calculate fit quality (diff bw all prices and predicted value)
    residuals_all = age_miles_predmiles['Miles'] - age_miles_predmiles['Predicted Miles']
    ss_res_all = np.sum(residuals_all**2)   # residual sum of squares
    ss_tot_all = np.sum((age_miles_predmiles['Miles'] - np.mean(age_miles_predmiles['Miles']))**2)   # total sum of squares
    r_squared_all = 1 - (ss_res_all / ss_tot_all)

    
    # plot fit
    plt.plot(x_data, lin_function(x_data, *popt), '#ff4c00', linewidth=3)
    
    # add textbox
    props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)    
    textbox_1 = r'Miles(Age) = $a{\bullet}$Age'
    textbox_2 = '$a$ = %5.0f' % (popt[0]) + '\n$R^{2}$ = %5.2f' % r_squared_all
    
    ax.text(0.035, 0.95, textbox_1, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_1)
    
    ax.text(0.035, 0.90, textbox_2, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)
    
    plt.tight_layout()
    
    # save figure
    if save == True:
        figure_name = '../images/Miles_vs_age.png'
        plt.savefig(figure_name, dpi = 600)
    else:
        pass

    
    
    
    
# plot depreciation curve - age
    
def plot_depr_age(data, model, newerthan, b_lower, counter, counts, alpha, save): # fit_data
    
    import numpy as np
    import pandas as pd
    
    # get model data from master list
    car_model_data_1 = data[data['Model'] == model]
    make = car_model_data_1['Make'].iloc[0]
    
    # filter data based on excluded years
    car_model_data_2 = car_model_data_1[car_model_data_1['Year'] > newerthan]
    
    # create dataframe with age and list price
    age_listprice = pd.DataFrame({'Age': 2020 - car_model_data_1['Year'],
                              'List Price': car_model_data_1['Price'],
                              })
    
    # group all listings by year, taking median value for curve fitting
    model_data_grouped_year = car_model_data_2.groupby('Year', as_index = False).mean() # group prices within year by median value
    
    # create x- and y- columns for fit
    year_age_median_price = pd.DataFrame({'Year': model_data_grouped_year['Year'],
                                'Age': 2020 - model_data_grouped_year['Year'],
                                'Median Price': model_data_grouped_year['Price']
                                })
    
    # fit data to function
    from scipy.optimize import curve_fit
    def exp_function(x, a, b):
        return a * np.exp(-b * x)
    
    popt, pcov = curve_fit(exp_function, year_age_median_price['Age'], 
                            year_age_median_price['Median Price'], 
                            absolute_sigma=False, maxfev=1000,
                            bounds=((10000, b_lower), (200000, 1)))
    
    print(popt[1])
    
    # create predicted list price vs. age
    price_predicted = pd.DataFrame({'Age': range(0,max(year_age_median_price['Age'])+1,1),
                                    'Predicted Price': exp_function(range(0,max(year_age_median_price['Age'])+1,1), 
                                                                    popt[0], popt[1]), 
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
    # plt.scatter(year_age_median_price['Age'], year_age_median_price['Median Price'])
    
    # set up figure axis
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.scatter(age_listprice['Age'], 
               age_listprice['List Price'], 
               edgecolor='black', facecolor='blue', alpha=alpha)
    
    # plot fit
    x_axis_smooth = np.arange(min(year_age_median_price['Age']), max(year_age_median_price['Age'])+1, .1)
    plt.plot(x_axis_smooth, exp_function(x_axis_smooth, *popt), '#ff4c00', linewidth=3)
    
    # set x- and y- labels
    xlabel = r'Age ($\it{t}$, years)'
    ylabel = r'Price ($\it{P}$, \$k)' 
    yscale = 1000

    plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylabel(ylabel, fontsize = 18, fontname = 'Helvetica')
    plt.title(str(make) + ' ' + str(model) + '  ($n$ = ' + str(counts) + ')', fontsize = 20, fontname = 'Helvetica')
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    
    # force integers on x-axis
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # set up text box
    props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    
    textbox_1 = r'$P(t) = a{\bullet}exp(-bt)$'
    textbox_2 = '$a$ = %5.0f \n$b$ = %0.3f' % (popt[0], popt[1]) + '\n$R^{2}$ = %5.2f' % r_squared_all
    textbox_3 = '$Half$ $life:$ ' + str(round(0.6931/popt[1], 2)) + ' $y$'
    
    ax.text(0.54, 0.95, textbox_1, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_1)
    
    ax.text(0.72, 0.85, textbox_2, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)
    
    ax.text(0.625, 0.65, textbox_3, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)
    
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/yscale))
    ax.yaxis.set_major_formatter(ticks)
    
    # save figure
    if save == True:
        figure_name = './images/depr_by_model/' + str(counter) + '_' + str(model) + '.png'
        plt.savefig(figure_name, dpi = 600)
    else:
        pass
    plt.show()

    # # return fit data
    # return(fit_data)


# plot depreciation curve - miles
    
def plot_depr_miles(data, model, newerthan, b_lower, counter, counts, alpha, save): # fit_data
    
    import numpy as np
    import pandas as pd
    
    # get model data from master list
    car_model_data_1 = data[data['Model'] == model]
    make = car_model_data_1['Make'].iloc[0]
    
    # filter data based on excluded years
    car_model_data_2 = car_model_data_1[car_model_data_1['Year'] > newerthan]
    
    # get rid of spurious listings: with list price under $10k and "zero miles"
    indexNames = car_model_data_2[(car_model_data_2['Mileage'] == 0) & (car_model_data_2['Price'] < 10000)].index
    car_model_data_2.drop(indexNames, inplace=True)
        
    # create dataframe with age and list price
    miles_listprice = pd.DataFrame({'Miles': car_model_data_2['Mileage'],
                                    'List Price': car_model_data_2['Price']})    
    
    # fit data to function
    from scipy.optimize import curve_fit
    def exp_function(x, a, b):
        return a * np.exp(-b * x)
    
    popt, pcov = curve_fit(exp_function, miles_listprice['Miles'], 
                            miles_listprice['List Price'], 
                            absolute_sigma=False, maxfev=1000,
                            bounds=((10000, b_lower), (200000, 1)))
    
    # create predicted list price vs. age
    price_predicted = pd.DataFrame({'Miles': range(0,max(miles_listprice['Miles'])+1,1),
                                    'Predicted Price': exp_function(range(0,max(miles_listprice['Miles'])+1,1), 
                                                                    popt[0], popt[1])})
    
    # combine list prices and predicted list prices
    miles_listprice_predprice = miles_listprice.merge(price_predicted, on='Miles', how='left')
    
    # calculate fit quality (diff bw all prices and predicted value)
    residuals_all = miles_listprice_predprice['List Price'] - miles_listprice_predprice['Predicted Price']
    ss_res_all = np.sum(residuals_all**2)   # residual sum of squares
    ss_tot_all = np.sum((miles_listprice_predprice['List Price'] - np.mean(miles_listprice_predprice['List Price']))**2)   # total sum of squares
    r_squared_all = 1 - (ss_res_all / ss_tot_all)
            
    # plot scatter data
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # plt.scatter(year_age_median_price['Age'], year_age_median_price['Median Price'])
    
    # set up figure axis
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.scatter(miles_listprice['Miles'], 
               miles_listprice['List Price'], 
               edgecolor='black', facecolor='blue', alpha=alpha)
    
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
    plt.title(str(make) + ' ' + str(model) + '  ($n$ = ' + str(counts) + ')', fontsize = 20, fontname = 'Helvetica')
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    
    # force integers on x-axis
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # set up text box
    props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    
    textbox_1 = r'$P(m) = a{\bullet}exp(-bm)$'
    textbox_2 = '$a$ = %5.0f \n$b$ = 1.34e-6' % (popt[0]) + '\n$R^{2}$ = %5.2f' % r_squared_all
    # textbox_3 = '$Half$ $life:$ ' + str(round(0.6931/popt[1], 2)) + ' $y$'
    
    ax.text(0.52, 0.95, textbox_1, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_1)
    
    ax.text(0.72, 0.85, textbox_2, transform = ax.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)
    
    plt.xlim(-10000,250000)    
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/xscale))
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/yscale))
    ax.xaxis.set_major_formatter(ticks)
    ax.yaxis.set_major_formatter(ticks)
    
    # save figure
    if save == True:
        figure_name = '../images/depr_by_mileage_civic.png'
        plt.savefig(figure_name, dpi = 600)
    else:
        pass
    plt.show()

    # # return fit data
    # return(fit_data)



# # plot depreciation curve - miles
    
# def plot_depr_miles(data, model, newerthan, counter, fit_data):
        
#     import pandas as pd
#     import numpy as np

#     # get model data from master list
#     car_model_data_1 = data[data['Model'] == model]
#     make = car_model_data_1['Make'].iloc[0]
    
#     # filter data based on excluded years
#     car_model_data_2 = car_model_data_1[car_model_data_1['Year'] > newerthan]
    
#     # create dataframe with age and list price
#     miles_listprice = pd.DataFrame({'Miles': car_model_data_2['Mileage'],
#                               'List Price': car_model_data_2['Price'],
#                               })   #.dropna()
    
        
#     # fit data to function
#     from scipy.optimize import curve_fit
#     def exp_function(x, a, b, c):
#         return a * np.exp(-b * x) + c
    
#     popt, pcov = curve_fit(exp_function, miles_listprice['Miles'], 
#                            miles_listprice['List Price'], 
#                            absolute_sigma=False, maxfev=1000,
#                            bounds=((10000, 0, 0), (200000, .003, 50000)))
    
#     # create predicted list price vs. miles 
#     price_predicted = pd.DataFrame({'Miles': range(0,max(miles_listprice['Miles'].astype(int))+1,1),
#                                     'Predicted Price': exp_function(range(0,max(miles_listprice['Miles'].astype(int))+1,1), 
#                                                                     popt[0], popt[1], popt[2]), 
#                                    })
    
#     # combine list prices and predicted list prices
#     miles_listprice_predprice = miles_listprice.merge(price_predicted, on='Miles', how='left')
    
#     # calculate fit quality (diff bw all prices and predicted value)
#     residuals_all = miles_listprice_predprice['List Price'] - miles_listprice_predprice['Predicted Price']
#     ss_res_all = np.sum(residuals_all**2)   # residual sum of squares
#     ss_tot_all = np.sum((miles_listprice_predprice['List Price'] - np.mean(miles_listprice_predprice['List Price']))**2)   # total sum of squares
#     r_squared_all = 1 - (ss_res_all / ss_tot_all)
    
#     # store fit data in dataframe
#     fit_data_temp = pd.DataFrame(data = [[counter, make, model, popt[0], popt[1], popt[2], r_squared_all]])
#     fit_data = pd.concat([fit_data, fit_data_temp], ignore_index=True)
        
#     # plot scatter data
#     import matplotlib.pyplot as plt
#     import matplotlib.ticker as ticker
#     # plt.scatter(miles_listprice['Miles'], miles_listprice['List Price'])
    
#     # set up figure axis
#     fig, ax = plt.subplots(1, 1, figsize=(7,7))
#     ax.scatter(miles_listprice['Miles'], 
#                miles_listprice['List Price'], 
#                edgecolor='black', facecolor='blue', alpha=0.1)
    
#     # plot fit
#     x_axis_smooth = np.arange(min(miles_listprice['Miles']), max(miles_listprice['Miles'])+1, .1)
#     plt.plot(x_axis_smooth, exp_function(x_axis_smooth, *popt), '#ff4c00', linewidth=3)
    
#     # set x- and y- labels
#     xlabel = r'Miles ($\it{m}$, k)'
#     ylabel = r'Price ($\it{P}$, \$k)'
#     xscale = 1000
#     yscale = 1000

#     plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
#     plt.ylabel(ylabel, fontsize = 18, fontname = 'Helvetica')
#     plt.title(str(make) + ' ' + str(model), fontsize = 20, fontname = 'Helvetica')
#     ax.tick_params(axis = 'x', labelsize = 14)
#     ax.tick_params(axis = 'y', labelsize = 14)
    
#     # # force integers on x-axis
#     # from matplotlib.ticker import MaxNLocator
#     # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
#     # get b in scientific notation
#     from decimal import Decimal
#     b_sci = '%.3E' % Decimal(popt[1])
    
#     # set up text box
#     props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
#     props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    
#     textbox_1 = r'$P(m) = a{\bullet}exp(-bm) + c$'
#     textbox_2 = '$a$ = %5.0f \n$b$ = %s \n$c$ =%5.0f' % (popt[0], float(b_sci), popt[2]) + '\n$R^{2}$ = %5.2f' % r_squared_all
    
#     ax.text(0.5, 0.95, textbox_1, transform = ax.transAxes, fontsize = 18, 
#             fontname = 'Helvetica', verticalalignment = 'top', bbox = props_1)
    
#     ax.text(0.72, 0.825, textbox_2, transform = ax.transAxes, fontsize = 18, 
#             fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)
    
#     for tick in ax.get_xticklabels():
#         tick.set_fontname('Helvetica')
#     for tick in ax.get_yticklabels():
#         tick.set_fontname('Helvetica')
    
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
#     # scale x- and y-axes
#     ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/xscale))
#     ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/yscale))
#     ax.xaxis.set_major_formatter(ticks)
#     ax.yaxis.set_major_formatter(ticks)
    
#     # # save figure, plot figure
#     # figure_name = '../images/depreciation/miles_depreciation_plots_by_model/' + str(counter) + '_' + str(model) + '.png'
#     # plt.savefig(figure_name, dpi = 600)
#     plt.show()

#     # # return fit data
#     # return(fit_data)


def plot_depr_R2(selection):
    
    # libraries
    import numpy as np
    import matplotlib.pyplot as plt
    
    # set up plot
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    plt.xlabel('Model', fontsize = 18, fontname = 'Helvetica')
    plt.ylabel('Fit quality ($R^2$)', fontsize = 18, fontname = 'Helvetica')
    plt.ylim(0,1.15)
    
    # set width of bar
    barWidth = 0.35
     
    # set height of bar
    
    bars_age = abs(selection['Fit_age_R2'])
    bars_miles = abs(selection['Fit_miles_R2'])
    # bars_cw = abs(selection['Fit_age_cw_R2'])
    
    # Set position of bar on x-axis
    r1 = np.arange(len(bars_age))
    r2 = [x + barWidth/2 for x in r1]
    r3 = [x + barWidth/1.25 for x in r2]
     
    # Make the plot
    plt.bar(r2, bars_age, color='blue', width=barWidth, edgecolor='white', label='Price ~ Age')
    plt.bar(r3, bars_miles, color='#ff4c00', width=barWidth, edgecolor='white', label='Price ~ Miles', zorder = 2)
    # plt.bar(r3, bars_cw, color='#ff4c00', width=barWidth, edgecolor='white', label='24%, 15%')
     
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth for r in range(len(bars_age))], 
               selection['Model'].tolist(), 
               rotation=45,
               ha='right')
    
    # adjust tick label size
    ax.tick_params(axis = 'x', labelsize = 18)
    ax.tick_params(axis = 'y', labelsize = 18)
     
    # Create legend & Show graphic
    plt.legend(prop={'size':15})
    # plt.title('Modeling depreciation', fontsize = 20, fontname = 'Helvetica')
    plt.tight_layout()
    plt.savefig('../images/R2_vs_model', dpi = 600)
    plt.show()
    


# plot histogram of car half life

def plot_hist_hl(data, user_choice_halflife, make_input, model_input, binwidth, textbox, props, xmin, xmax, xlabel, ylabel, figure_name):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    bins = np.arange(round(min(data),1) - 0.5*binwidth, max(data) + binwidth - 0.5*binwidth, binwidth)
    props = dict(facecolor='white', alpha=1.0)

    # n, bins, patches = 
    ax.hist(data, bins, edgecolor = 'black', facecolor = 'blue')
    
    plt.xlim(xmin, xmax); plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    # plt.ylim(ymin, ymax)
    plt.ylabel(ylabel, fontsize = 18)
    ax.tick_params(axis = 'x', labelsize = 14); ax.tick_params(axis = 'y', labelsize = 14)
    
    # force integers on y-axis
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
        
    ax.text(0.95, 0.95, textbox, 
            transform = ax.transAxes, 
            fontsize = 18, 
            fontname = 'Helvetica', 
            horizontalalignment = 'right',
            verticalalignment = 'top', 
            bbox = props)

    ax.axvline(x=user_choice_halflife, ymin=0, ymax=.125, linewidth=4, color='r')
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    plt.title(str(make_input) + ' ' + str(model_input), fontsize=20)
    
    # patches[3].set_fc('r')
    
    plt.tight_layout()

    plt.savefig(figure_name, dpi = 600)
    plt.show()


