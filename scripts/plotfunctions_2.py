#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:33:02 2020

@author: michaelboles
"""

# plot textbox formatted for three-parameter fit: P(t) = a*exp(-bt)+c

def plot_depr_age_segment(model, emp_data, fit_data_filtered, pred_data, save=False):
    
    # plots empirical data of given model with fits from segment 
    
    import pandas as pd
    
    make = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Make'].iloc[0]
    emp_data_model = emp_data[['Age', str(model)]]
    
    # get top ten cars (by counts) within segment
    vehicle_segment = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Body'].iloc[0]
    segment_topten = fit_data_filtered[fit_data_filtered['Body'] == str(vehicle_segment)][:10]
    
    # for each vehicle within segment, pull in price predictions (based on age)
    pred_price_segment = pd.DataFrame({str(segment_topten.iloc[0][2]): pred_data[str(segment_topten.iloc[0][2])],
                                       str(segment_topten.iloc[1][2]): pred_data[str(segment_topten.iloc[1][2])],
                                       str(segment_topten.iloc[2][2]): pred_data[str(segment_topten.iloc[2][2])],
                                       str(segment_topten.iloc[3][2]): pred_data[str(segment_topten.iloc[3][2])],
                                       str(segment_topten.iloc[4][2]): pred_data[str(segment_topten.iloc[4][2])],
                                       str(segment_topten.iloc[5][2]): pred_data[str(segment_topten.iloc[5][2])],
                                       str(segment_topten.iloc[6][2]): pred_data[str(segment_topten.iloc[6][2])],
                                       str(segment_topten.iloc[7][2]): pred_data[str(segment_topten.iloc[7][2])],
                                       str(segment_topten.iloc[8][2]): pred_data[str(segment_topten.iloc[8][2])],
                                       str(segment_topten.iloc[9][2]): pred_data[str(segment_topten.iloc[9][2])]})
    
    # normalize empirical prices for selected model and fit prices for other models in segment
    
    # if no empirical new price, insert fitted new price as best guess
    import math
    if math.isnan(emp_data_model.iloc[0][1]):
        emp_data_model.at[0,str(model)] = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Fit_age_a'].iloc[0] + fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Fit_age_c'].iloc[0]
    else:
        pass
        
        
    fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Fit_age_a'].iloc[0] + fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Fit_age_c'].iloc[0]
    
    
    emp_data_model_norm = pd.DataFrame({'Age': emp_data_model['Age'],
                                        'Norm price': emp_data_model[str(model)]/emp_data_model[str(model)][0]})
    
    pred_price_segment_norm = pd.DataFrame({str(segment_topten.iloc[0][2]): pred_price_segment.iloc[:,0]/pred_price_segment.iloc[0][0],
                                       str(segment_topten.iloc[1][2]): pred_price_segment.iloc[:,1]/pred_price_segment.iloc[0][1],
                                       str(segment_topten.iloc[2][2]): pred_price_segment.iloc[:,2]/pred_price_segment.iloc[0][2],
                                       str(segment_topten.iloc[3][2]): pred_price_segment.iloc[:,3]/pred_price_segment.iloc[0][3],
                                       str(segment_topten.iloc[4][2]): pred_price_segment.iloc[:,4]/pred_price_segment.iloc[0][4],
                                       str(segment_topten.iloc[5][2]): pred_price_segment.iloc[:,5]/pred_price_segment.iloc[0][5],
                                       str(segment_topten.iloc[6][2]): pred_price_segment.iloc[:,6]/pred_price_segment.iloc[0][6],
                                       str(segment_topten.iloc[7][2]): pred_price_segment.iloc[:,7]/pred_price_segment.iloc[0][7],
                                       str(segment_topten.iloc[8][2]): pred_price_segment.iloc[:,8]/pred_price_segment.iloc[0][8],
                                       str(segment_topten.iloc[9][2]): pred_price_segment.iloc[:,9]/pred_price_segment.iloc[0][9]})
    
    
    # plot empirical median price for selected model
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    plt.plot(emp_data_model_norm['Age'], emp_data_model_norm['Norm price'], linewidth=2)
    
    # plot empirical prices for other models
    plt.plot(pred_price_segment_norm.index, pred_price_segment_norm.iloc[:,1:10], color=[0.75, 0.75, 0.75], zorder=0)
    
    # set x- and y- labels
    xlabel = 'Age (years)'
    ylabel = 'Price (normalized)' 
    
    plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylabel(ylabel, fontsize = 18, fontname = 'Helvetica')
    plt.title(str(make) + ' ' + str(model), fontsize = 20, fontname = 'Helvetica')
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    
    # force integers on x-axis
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # plt.legend()

    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
    
    # # save figure
    # if save == True:
    #     figure_name = '../images/depreciation/age_depreciation_plots_by_model/' + str(counter) + '_' + str(model) + '.png'
    #     plt.savefig(figure_name, dpi = 600)
    # else:
    #     pass
        
    plt.show()
    

def plot_depr_age_segment2(model, fit_data_filtered, pred_data, save=False):
    
    # plots fit data of given model with fits from segment 
    
    import pandas as pd
    
    make = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Make'].iloc[0]
    
    # get top ten cars (by counts) within segment
    vehicle_segment = fit_data_filtered[fit_data_filtered['Model'] == str(model)]['Body'].iloc[0]
    segment_topten = fit_data_filtered[fit_data_filtered['Body'] == str(vehicle_segment)][:10]
    
    # for each vehicle within segment, pull in price predictions (based on age)
    pred_price_segment = pd.DataFrame({str(segment_topten.iloc[0][2]): pred_data[str(segment_topten.iloc[0][2])],
                                       str(segment_topten.iloc[1][2]): pred_data[str(segment_topten.iloc[1][2])],
                                       str(segment_topten.iloc[2][2]): pred_data[str(segment_topten.iloc[2][2])],
                                       str(segment_topten.iloc[3][2]): pred_data[str(segment_topten.iloc[3][2])],
                                       str(segment_topten.iloc[4][2]): pred_data[str(segment_topten.iloc[4][2])],
                                       str(segment_topten.iloc[5][2]): pred_data[str(segment_topten.iloc[5][2])],
                                       str(segment_topten.iloc[6][2]): pred_data[str(segment_topten.iloc[6][2])],
                                       str(segment_topten.iloc[7][2]): pred_data[str(segment_topten.iloc[7][2])],
                                       str(segment_topten.iloc[8][2]): pred_data[str(segment_topten.iloc[8][2])],
                                       str(segment_topten.iloc[9][2]): pred_data[str(segment_topten.iloc[9][2])]})
    
    # normalize empirical prices for selected model and fit prices for other models in segment
    
    pred_price_modeled_norm = pd.DataFrame({'Age': pred_data.index,
                                            'Norm price': pred_data[str(model)]/pred_data[str(model)][0]})
    
    pred_price_segment_norm = pd.DataFrame({str(segment_topten.iloc[0][2]): pred_price_segment.iloc[:,0]/pred_price_segment.iloc[0][0],
                                       str(segment_topten.iloc[1][2]): pred_price_segment.iloc[:,1]/pred_price_segment.iloc[0][1],
                                       str(segment_topten.iloc[2][2]): pred_price_segment.iloc[:,2]/pred_price_segment.iloc[0][2],
                                       str(segment_topten.iloc[3][2]): pred_price_segment.iloc[:,3]/pred_price_segment.iloc[0][3],
                                       str(segment_topten.iloc[4][2]): pred_price_segment.iloc[:,4]/pred_price_segment.iloc[0][4],
                                       str(segment_topten.iloc[5][2]): pred_price_segment.iloc[:,5]/pred_price_segment.iloc[0][5],
                                       str(segment_topten.iloc[6][2]): pred_price_segment.iloc[:,6]/pred_price_segment.iloc[0][6],
                                       str(segment_topten.iloc[7][2]): pred_price_segment.iloc[:,7]/pred_price_segment.iloc[0][7],
                                       str(segment_topten.iloc[8][2]): pred_price_segment.iloc[:,8]/pred_price_segment.iloc[0][8],
                                       str(segment_topten.iloc[9][2]): pred_price_segment.iloc[:,9]/pred_price_segment.iloc[0][9]})
    
    # plot empirical median price for selected model
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    plt.plot(pred_price_modeled_norm['Age'], 
             pred_price_modeled_norm['Norm price'], 
             linewidth=2, 
             label=str(make) + ' ' + str(model))
    
    # plot empirical prices for other models
    plt.plot(pred_price_segment_norm.index, 
             pred_price_segment_norm.iloc[:,1:10], 
             color=[0.75, 0.75, 0.75], 
             label='Other ' + str(vehicle_segment) + 's',
             zorder=0)
    
    # set x- and y- labels
    xlabel = 'Age (years)'
    ylabel = 'Price (normalized)' 
    
    plt.xlabel(xlabel, fontsize = 18, fontname = 'Helvetica')
    plt.ylabel(ylabel, fontsize = 18, fontname = 'Helvetica')
    plt.title(str(make) + ' ' + str(model), fontsize = 20, fontname = 'Helvetica')
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    
    # force integers on x-axis
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], prop={'size': 14})
    
    # plt.legend()

    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
    
    # # save figure
    # if save == True:
    #     figure_name = '../images/depreciation/age_depreciation_plots_by_model/' + str(counter) + '_' + str(model) + '.png'
    #     plt.savefig(figure_name, dpi = 600)
    # else:
    #     pass
        
    plt.show()

