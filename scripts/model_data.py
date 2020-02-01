#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:36:20 2020

@author: michaelboles
"""


# set up working directory
import os
os.chdir('/Users/michaelboles/Michael/Coding/2020/Insight/Project/Cars/scripts') 

# open full listings data set and depreciation summary
import pandas as pd
listings = pd.read_csv('../data/listings5.csv')
depr_summary = pd.read_csv('../data/depreciation/depreciation_by_model_2.csv')


# ### PRE-PROCESS DATA ###


# add location column
location = listings['Description'].str.split('.', 1, expand=True)[0]
location_2 = location.str.split(': ', 1, expand=True)
listings['Location'] = location_2[1]

# drop unwanted columns
listings_2 = listings.drop(['ListTitle','URL', 'VIN'], axis=1)

# check dataframe
listings_short = listings_2[:1000]
listings_2.columns

# pull out numerical columns
listings_2_numerical = listings_2[['Price', 'Year', 'Mileage']]


### encode categorical variables

body_dummies = listings_2['Body'].str.get_dummies().drop(columns='Unavailable ')
color_dummies = listings_2['Color'].str.get_dummies()
drive_dummies = listings_2['Drive'].str.get_dummies()
drive_dummies.sum()
transmission_dummies = listings_2['Transmission'].str.get_dummies()
location_dummies = listings_2['Location'].str.get_dummies().drop(columns = ['Corte Madera, CA', 
                                                                                        'Costa Mesa, CA', 
                                                                                        'Skokie, IL', 
                                                                                      'Long Island City, NY'])
model_dummies = listings_2['Model'].str.get_dummies()


listings_dummies = pd.concat([listings_2_numerical, 
                              # body_dummies, 
                              # color_dummies,
                              drive_dummies,
                              transmission_dummies,
                              location_dummies,
                              model_dummies], 
                              axis=1)

# save dummified listings
listings_dummies.to_csv('../data/valuation/listings_dummies.csv', index=False, compression='gzip')






### MODEL DATA ###

# import data
import pandas as pd
listings_dummies = pd.read_csv('../data/valuation/listings_dummies.csv', compression='gzip')
listings_dummies_short = listings_dummies #[:10000]
columns = listings_dummies_short.columns

# set features (independent) and labels (dependent)
X = listings_dummies_short.drop(columns='Price')
y = listings_dummies_short['Price']




### STATSMODELS ###

# split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# fit data - Statsmodels
import statsmodels.api as sm
model = sm.OLS(y_train, X_train)

# get training set summary
results_train = model.fit()
summary_train = results_train.summary()
summary_train_text = summary_train.as_text()
pvalues_train = results_train.pvalues

# predict test set
y_pred = results_train.predict(X_test)


summary_test = results_test














# ### SCIKIT-LEARN ###

# from sklearn.linear_model import LinearRegression
# from sklearn.feature_selection import RFECV

# # with with linear regression
# lr = LinearRegression()
# rfecv = RFECV(estimator=lr, step=1, cv=5, scoring='r2')
# rfecv.fit(X, y)
# # grid_scores = rfecv.grid_scores_
# # print(rfecv.grid_scores_)

# features_selected = [f for f,s in zip(X.columns, rfecv.support_) if s]

# print("Optimal number of features : %d" % rfecv.n_features_)

# # Plot number of features VS. cross-validation scores
# import matplotlib.pyplot as plt
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

# # Put the best features into new dataframe
# X_new = rfecv.transform(X)



# fit data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
regressor.intercept_, regressor.coef_



# leave one out cross-validation
from sklearn.model_selection import LeaveOneOut 
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test, y_train, y_test)



# predict values based on model
y_pred = regressor.predict(x) # listing + surroundings data
y_pred2 = regressor.predict(x2) # listing data only

# calculate difference between predicted and actual prices
diff = round((y - y_pred2), 6)

# add difference to full data set
data_all['Price difference 2'] = diff
#data_all.to_csv('data_all_price_predictions.csv')













