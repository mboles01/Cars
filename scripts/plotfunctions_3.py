#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:33:02 2020

@author: michaelboles
"""




# plot textbox formatted for two-parameter fit: P(t) = a*exp(-bt), with two subplots

    
def plot_combo_depr(data, fit_data_filtered, pred_data, model, newerthan, counter, save): 
    
    import numpy as np
    import pandas as pd
    
    ### FIRST PLOT
    
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
    def exp_function(x, a, b):
        return a * np.exp(-b * x)
    
    popt, pcov = curve_fit(exp_function, year_age_median_price['Age'], 
                            year_age_median_price['Median Price'], 
                            absolute_sigma=False, maxfev=1000,
                            bounds=((10000, 0.05), (200000, 1)))
    
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
            
    # plot scatter data
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    # set up figure axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    ax1.scatter(age_listprice['Age'], 
               age_listprice['List Price'], 
               edgecolor='black', facecolor='blue', alpha=0.33)
    
    # plot fit
    x_axis_smooth = np.arange(min(year_age_median_price['Age']), max(year_age_median_price['Age'])+1, .1)
    ax1.plot(x_axis_smooth, exp_function(x_axis_smooth, *popt), '#ff4c00', linewidth=3)
    
    # set x- and y- labels
    xlabel = r'Age ($\it{t}$, years)'
    ylabel = r'Price ($\it{P}$, \$k)' 
    yscale = 1000
    
    fig.suptitle(str(make) + ' ' + str(model), fontsize = 24, fontname = 'Helvetica')
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel(ylabel, fontsize=18)
    
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    
    # force integers on x-axis
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # set up text box
    props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    props_2 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    
    textbox_1 = r'$P(t) = a{\bullet}exp(-bt)$'
    textbox_2 = '$a$ = %5.0f \n$b$ = %0.3f' % (popt[0], popt[1]) + '\n$R^{2}$ = %5.2f' % r_squared_all
    
    ax1.text(0.4, 0.95, textbox_1, transform = ax1.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_1)
    
    ax1.text(0.65, 0.825, textbox_2, transform = ax1.transAxes, fontsize = 18, 
            fontname = 'Helvetica', verticalalignment = 'top', bbox = props_2)
    
    for tick in ax1.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax1.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    # ax1.rcParams['axes.unicode_minus'] = False
    ax1.grid(); ax1.grid(color=(.9, .9, .9)); ax1.set_axisbelow(True)
    
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/yscale))
    ax1.yaxis.set_major_formatter(ticks)
    
    
    ### SECOND PLOT
    
    # plots fit data of given model with fits from segment 
    
    make = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Make'].iloc[0]
    
    # get top ten cars (by counts) within segment
    vehicle_segment = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Body'].iloc[0]
    segment_topten = fit_data_filtered[fit_data_filtered['Body'] == str(vehicle_segment)][:10]
    
    # for each vehicle within segment, pull in price predictions (based on age)
    columns = segment_topten[:10].iloc[:,2].values.tolist()
    
    pred_price_segment = pd.DataFrame(columns=columns)
    pred_price_segment_norm = pd.DataFrame(columns = columns)

    for i in range(len(columns)):
        pred_price_segment[str(columns[i])] = pred_data[str(columns[i])]
        pred_price_segment_norm[str(columns[i])] = pred_data[str(columns[i])]/pred_data[str(columns[i])][0]
          
    # normalize empirical prices for selected model and fit prices for other models in segment
    
    pred_price_modeled_norm = pd.DataFrame({'Age': pred_data.index,
                                            'Norm price': pred_data[str(model)]/pred_data[str(model)][0]})
        
    # plot empirical median price for selected model
    ax2.plot(pred_price_modeled_norm['Age'], 
             pred_price_modeled_norm['Norm price'], 
             linewidth=2, 
             label=str(make) + ' ' + str(model))
    
    # plot empirical prices for other models
    ax2.plot(pred_price_segment_norm.index, 
             pred_price_segment_norm.iloc[:,1:10], 
             color=[0.75, 0.75, 0.75], 
             label='Other ' + str(vehicle_segment) + 's',
             zorder=1)
    
    # set x- and y- labels
    xlabel = 'Age (years)'
    ylabel = 'Price (normalized)' 
    
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel(ylabel, fontsize=18)
    
    ax2.tick_params(axis = 'x', labelsize = 14)
    ax2.tick_params(axis = 'y', labelsize = 14)
    
    # force integers on x-axis
    from matplotlib.ticker import MaxNLocator
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # add legend
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], labels[:2], prop={'size': 14})
    
    # plt.legend()

    for tick in ax2.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax2.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    # ax2.rcParams['axes.unicode_minus'] = False
    ax2.grid(); ax2.grid(color=(.9, .9, .9)); ax2.set_axisbelow(True)
    
    
    # save figure
    if save == True:
        figure_name = '../images/depreciation/depr_by_model/' + str(counter) + '_' + str(model) + '.png'
        plt.savefig(figure_name, dpi = 600)
    else:
        pass
    fig.tight_layout(pad=5)
    plt.show()

