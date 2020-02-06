#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:30:31 2020

@author: michaelboles
"""

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Cars/scripts') 

# open full listings data set and depreciation summary
import pandas as pd
listings_data = pd.read_csv('../data/listings5_filtered.csv')
depr_summary = pd.read_csv('../data/depr_summary_filtered.csv')


### HISTOGRAM PLOT: PRICE ###
import numpy as np
data = depr_summary['Half life']

# textbox
average = round(np.nanmean(data), 2)
median = round(np.nanmedian(data), 2)
stdev = round(np.std(data), 2)
props = dict(facecolor='white', edgecolor='none', alpha=0.67, boxstyle='square, pad=1')
textbox = '$Half$ $life$ $(years)$\nAverage = %.1f \nMedian = %.1f \nStdev = %.1f' % (average, median, stdev)

import numpy as np
binwidth = 0.5
xmin = 1
xmax = 11
ymin = 0
ymax = 35
xlabel = 'Half life (years)'
ylabel = 'Counts (number of car models)'
figure_name = '../images/Depreciation_hist.png'

from plotfunctions import plot_hist
plot_hist(data, binwidth, textbox, props, xmin, xmax, ymin, ymax, xlabel, ylabel, figure_name)




### Box plots ###

# Box plot: depreciation across brands

# import data - depreciation across models
depr_summary = pd.read_csv('../data/depreciation/depreciation_by_model.csv')
depr_summary = pd.read_csv('../data/depr_summary_filtered.csv')

# remove some brands
indexNames = depr_summary[(depr_summary['Make'] == 'Maserati') | 
                          (depr_summary['Make'] == 'Jaguar') |
                          (depr_summary['Make'] == 'smart') |
                          (depr_summary['Make'] == 'Mitsubishi')].index

depr_summary.drop(indexNames, inplace=True)

# rename some brands
depr_summary.loc[(depr_summary['Make'] == 'MINI'),'Make'] = 'Mini'
depr_summary.loc[(depr_summary['Make'] == 'MAZDA'),'Make'] = 'Mazda'
depr_summary.loc[(depr_summary['Make'] == 'INFINITI'),'Make'] = 'Infinity'


# Determine order
depr_order_brand = depr_summary.groupby('Make').median().sort_values(by='Half life',ascending=True)

# create seaborn box + strip plot
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (45,22))

ax = sns.boxplot(x = 'Make', y = 'Half life', data = depr_summary, 
                 showfliers = False, order = list(depr_order_brand.index), linewidth = 5)
ax = sns.stripplot(x = 'Make', y = 'Half life', data = depr_summary,
                 order = list(depr_order_brand.index), jitter = 0.25, size = 15,
                 linewidth = 3, edgecolor = 'black', alpha = 0.5)

# set axis properties
plt.xticks(rotation=45, fontname = 'Helvetica', fontsize = 42, ha = 'right')
plt.yticks(fontname = 'Helvetica', fontsize = 42)


plt.xlabel('Car make', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Half life (years)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(2, 8); ax.yaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3)

figure_name = '../images/depr_across_brands_R2>0.67.png'

plt.tight_layout()
plt.savefig(figure_name, dpi = 150)




# Box plot: depreciation across segments

# Determine order
depr_order_brand = depr_summary.groupby('Body').median().sort_values(by='Half life',ascending=True)


# create seaborn box + strip plot
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (14,14))

ax = sns.boxplot(x = 'Body', 
                 y = 'Half life', 
                 data = depr_summary,
                 showfliers = False, 
                 order = list(depr_order_brand.index), 
                 linewidth = 5)

ax = sns.stripplot(x = 'Body', 
                   y = 'Half life', 
                   data = depr_summary,
                   order = list(depr_order_brand.index), 
                   jitter = 0.25, size = 15,
                   linewidth = 3, edgecolor = 'black', alpha = 0.5)

# set axis properties
import numpy as np
plt.xticks(np.arange(5), ('SUV', 'Sedan', 'Van', 'Coupe', 'Truck'),
    rotation=45, fontname = 'Helvetica', fontsize = 42, ha = 'right')
plt.yticks(fontname = 'Helvetica', fontsize = 42)
# plt.xticks(np.arange(5), ('SUV', 'Sedan', 'Van', 'Coupe', 'Truck'))

plt.xlabel('Body type', fontsize = 55, fontname = 'Arial', fontweight = 'bold')
plt.ylabel('Half life (years)', fontsize = 55, fontname = 'Arial', 
           fontweight = 'bold')

ax.set_ylim(0, 10); ax.yaxis.labelpad = 25
ax.xaxis.set_tick_params(width = 3, length = 15)
ax.yaxis.set_tick_params(width = 3, length = 15)
plt.setp(ax.spines.values(), linewidth = 3)

figure_name = '../images/depr_across_segment_R2>0.67.png'

plt.tight_layout()

plt.savefig(figure_name, dpi = 150)

plt.show()




### FIT SURFACE ###

# # collect top n models by count frequency
listings_sorted = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)

# select model and pull out data
model = 'Civic'
model_data = listings_data[listings_data['Model'] == str(model)]

x = 2020 - model_data['Year']
y = model_data['Mileage']
z = model_data['Price']


# create x- and y- columns for fit
price_age_miles = pd.DataFrame({'Age': x,
                               'Mileage': y,
                               'Price': z})

def exp_function_2(x, a, b, c):
    return (a/2) * (np.exp(-b * x['Age']) + np.exp(-c * x['Mileage']))

popt, pcov = curve_fit(exp_function_2, 
                       price_age_miles,                     # x, y-values
                       price_age_miles['Price'],            # z-values
                       absolute_sigma=False, maxfev=10000,
                       bounds=((10000, 0.1, 0.0000001), (200000, 1, 0.00001)))

age_max = 20
mileage_max = 250000
price_max = 30000

# filter (x, y, z) data
x_filtered = 2020 - model_data[(model_data['Mileage'] < mileage_max) & 
                               (2020 - model_data['Year'] < age_max) & 
                               (model_data['Price'] < price_max)]['Year']

y_filtered = model_data[(model_data['Mileage'] < mileage_max) & 
                        (2020 - model_data['Year'] < age_max) & 
                        (model_data['Price'] < price_max)]['Mileage']

z_filtered = model_data[(model_data['Mileage'] < mileage_max) & 
                        (2020 - model_data['Year'] < age_max) & 
                        (model_data['Price'] < price_max)]['Price']

# create predicted values - surface
x_pred = range(0, age_max + 1, 1)
y_pred = range(0, mileage_max + 1, 1000)
X, Y = np.meshgrid(x_pred, y_pred)

def exp_function_2_2(x, y, a, b, c):
    return (a/2) * (np.exp(-b * x) + np.exp(-c * y))

zs = np.array(exp_function_2_2(np.ravel(X), np.ravel(Y), popt[0], popt[1], popt[2]))
Z = zs.reshape(X.shape)



v_angles_1 = [item for item in range(184,264,2)]
v_angles_2 = [item for item in range(264,183,-2)]
v_angles = v_angles_1 + v_angles_2

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

ims = []
for angle in v_angles:
    print(angle)
    fig = plt.figure(figsize = (13,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, Z, 
                    cmap=plt.cm.coolwarm, 
                    alpha=0.67, 
                    edgecolor='white', 
                    linewidth=0.25, 
                    zorder=-1)
    
    ax.scatter(x_filtered, 
               y_filtered, 
               z_filtered, 
               alpha=0.25, 
               lw=0.25, 
               facecolor='blue',
               edgecolor='black',
               zorder=1)
    
    r_squared_all = 0.95
    counts = 1484
    
    # # set up text box
    # props_1 = dict(facecolor='white', edgecolor='none', alpha=0.67)
    
    
    # textbox_1 = r'$P(t,m) = (a/2){\bullet}[exp(-bt)+exp(-cm)]$'
    # textbox_2 = '$a$ = %5.0f \n$b$ = %0.3f \n$c$ = 9.1e-6' % (popt[0], popt[1]) + '\n$R^{2}$ = %5.2f' % r_squared_all + '\n$n$ = %d' % counts
    
    # ax.text(-15, 0, 41000,
    #           s=textbox_1, 
    #           fontsize = 12, 
    #           fontname = 'Helvetica', 
    #           horizontalalignment = 'left',
    #           verticalalignment = 'top', 
    #           bbox = props_1)
    
    # ax.text(-15, 0, 35000,
    #           s=textbox_2, 
    #           fontsize = 12, 
    #           fontname = 'Helvetica', 
    #           horizontalalignment = 'left',
    #           verticalalignment = 'top', 
    #           bbox = props_1)

    # ax.text(0, 0, 31000,
    #           s='Honda Civic', 
    #           fontsize = 18, 
    #           fontname = 'Helvetica', 
    #           horizontalalignment = 'left',
    #           verticalalignment = 'top', 
    #           bbox = props_1)
    
    ax.axes.set_xlim3d(left=0, right=20) 
    ax.axes.set_ylim3d(top=0, bottom=250000) 
    ax.axes.set_zlim3d(bottom=0, top=30001) 
    
    x_start, x_end = ax.get_xlim()
    z_start, z_end = ax.get_zlim()
    ax.xaxis.set_ticks(np.arange(x_start, x_end, 5))
    ax.zaxis.set_ticks(np.arange(z_start, z_end, 10000))
    
    plt.xlabel(xlabel, fontsize = 16, fontname = 'Helvetica')
    plt.ylabel(ylabel, fontsize = 16, fontname = 'Helvetica')
    # plt.zlabel(zlabel, fontsize = 16, fontname = 'Helvetica')
    
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.tick_params(axis = 'z', labelsize = 14)
    
    
    ax.set_xlabel('\n\nAge ($t$, years)')
    ax.set_ylabel('\n\n      Mileage ($m$, k miles)')
    ax.set_zlabel('\n\nPrice ($k)', fontsize = 16, fontname = 'Helvetica')
    
    plt.grid(); ax.grid(color=(.9, .9, .9)); ax.set_axisbelow(True)
    
    import matplotlib.ticker as ticker
    ticks = ticker.FuncFormatter(lambda y_filtered, pos: '{0:g}'.format(y_filtered/1000))
    ticks = ticker.FuncFormatter(lambda z_filtered, pos: '{0:g}'.format(z_filtered/1000))
    ax.yaxis.set_major_formatter(ticks)
    ax.zaxis.set_major_formatter(ticks)
    
    ax.view_init(15, angle)
    ax.dist = 11
    
    im = plt.gcf()
    
    plt.show()
    ims.append([im])
        
ani = animation.ArtistAnimation(fig, 
                                ims,
                                interval=50,
                                blit=True,
                                repeat_delay=500)

writer = PillowWriter(fps=20)
ani.save("3d_scatter.gif", writer='imagemagick')

        # plt.savefig('../images/gif2/' + str(v_angle) + '.png', dpi = 100)
        # plt.show()    




# # calculate fit quality (diff bw all prices and predicted value)
# residuals_all = age_miles_predmiles['Miles'] - age_miles_predmiles['Predicted Miles']
# ss_res_all = np.sum(residuals_all**2)   # residual sum of squares
# ss_tot_all = np.sum((age_miles_predmiles['Miles'] - np.mean(age_miles_predmiles['Miles']))**2)   # total sum of squares
# r_squared_all = 1 - (ss_res_all / ss_tot_all)

    
    
    

