#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:49:05 2020

@author: michaelboles
"""


import matplotlib.pyplot as plt
import numpy as np
print('numpy: '+np.version.full_version)
import matplotlib.animation as animation
import matplotlib
print('matplotlib: '+matplotlib.__version__)

Nfrm = 10
fps = 10

def generate(X, Y, phi):
    '''
    Generates Z data for the points in the X, Y meshgrid and parameter phi.
    '''
    R = 1 - np.sqrt(X**2 + Y**2)
    return np.cos(2 * np.pi * X + phi) * R


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Make the X, Y meshgrid.
xs = np.linspace(-1, 1, 50)
ys = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(xs, ys)

# Set the z axis limits so they aren't recalculated each frame.
ax.set_zlim(-1, 1)

# Begin plotting.
wframe = None
Z = generate(X, Y, 0)
def update(idx):
    phi=phis[idx]
    global wframe
    # If a line collection is already remove it before drawing.
    if wframe:
        ax.collections.remove(wframe)

    # Plot the new wireframe and pause briefly before continuing.
    Z = generate(X, Y, phi)
    wframe = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='k', linewidth=0.5)
phis = np.linspace(0, 180. / np.pi, 100)
ani = animation.FuncAnimation(fig, update, Nfrm, interval=1000/fps)


fn = '../images/plot_wireframe_funcanimation'
ani.save(fn+'.gif',writer='imagemagick',fps=fps)








### FIT SURFACE ###

# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Cars/scripts') 

# open full listings data set and depreciation summary
import pandas as pd
listings_data = pd.read_csv('../data/listings5_filtered.csv')
depr_summary = pd.read_csv('../data/depr_summary_filtered.csv')


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

from scipy.optimize import curve_fit
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



fig = plt.figure(figsize = (13,8))
ax = fig.add_subplot(111, projection='3d')




# Begin plotting
wframe = None
i = 0
def update(v_angles):
    angle = v_angles[i]
    global wframe
    # If a line collection is already remove it before drawing.
    if wframe:
        ax.collections.remove(wframe)
    wframe = ax.plot_surface(X, Y, Z, 
                cmap=plt.cm.coolwarm, 
                alpha=0.67, 
                edgecolor='white', 
                linewidth=0.25, 
                zorder=-1)
    ax.view_init(15, angle)
    i += 1
    ax.dist = 11
    

# # Plot the new wireframe and pause briefly before continuing.
# wframe = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='k', linewidth=0.5)
# phis = np.linspace(0, 180. / np.pi, 100)
# ani = animation.FuncAnimation(fig, update, Nfrm, interval=1000/fps)


# fn = '../images/plot_wireframe_funcanimation'
# ani.save(fn+'.gif',writer='imagemagick',fps=fps)


# ax.scatter(x_filtered, 
#            y_filtered, 
#            z_filtered, 
#            alpha=0.25, 
#            lw=0.25, 
#            facecolor='blue',
#            edgecolor='black',
#            zorder=1)

# wframe = plt.gcf()


v_angles = [item for item in range(184,264,2)] + [item for item in range(264,183,-2)]

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

ani = animation.FuncAnimation(fig, update, 10, interval=1000/10)
fn = '../images/3d_scatter'
ani.save(fn+'.gif',writer='imagemagick',fps=fps)








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




