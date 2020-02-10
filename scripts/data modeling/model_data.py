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
model_counts = pd.read_csv('../data/make_model_list_sorted.csv')


### PRE-PROCESS DATA ###

# # add location column
# location = listings['Description'].str.split('.', 1, expand=True)[0]
# location_2 = location.str.split(': ', 1, expand=True)
# listings['Location'] = location_2[1]

# # drop unwanted columns
# listings_2 = listings.drop(['ListTitle','URL', 'VIN'], axis=1)

# # filter out models with fewer than 100 observations
# selection = pd.DataFrame({'Model': model_counts[model_counts['Count'] > 100]['Model']})
# listings_2_filtered = listings_2.merge(selection.reset_index(), on='Model', how='right').drop(columns='index')

# # pull out numerical columns
# listings_2_numerical = listings_2_filtered[['Price', 'Year', 'Mileage']]
# listings_2_selected = listings_2_filtered[['Price', 'Year', 'Mileage', 'Make']]

# # add age column, remove year column
# listings_2_numerical['Age'] = 2020 - listings_2_numerical['Year']
# listings_2_numerical = listings_2_numerical.drop(columns=['Year'])
# listings_2_selected['Age'] = 2020 - listings_2_selected['Year']
# listings_2_selected = listings_2_selected.drop(columns=['Year'])


# ### encode categorical variables

# body_dummies = listings_2_filtered['Body'].str.get_dummies().drop(columns='Unavailable ')
# color_dummies = listings_2_filtered['Color'].str.get_dummies()
# drive_dummies = listings_2_filtered['Drive'].str.get_dummies()
# transmission_dummies = listings_2_filtered['Transmission'].str.get_dummies()
# location_dummies = listings_2_filtered['Location'].str.get_dummies().drop(columns = ['Corte Madera, CA', 
#                                                                                         'Costa Mesa, CA', 
#                                                                                         'Skokie, IL', 
#                                                                                       'Long Island City, NY'])
# model_dummies = listings_2_filtered['Model'].str.get_dummies()


# listings_dummies = pd.concat([listings_2_numerical, 
#                               # body_dummies, 
#                               # color_dummies,
#                               drive_dummies,
#                               transmission_dummies,
#                               location_dummies,
#                               model_dummies], 
#                               axis=1)

# # save filtered (no dummy) listings
# listings_2_selected.to_csv('../data/valuation/listings_make_filtered.csv', index=False, compression='gzip')

# # save dummified listings
# listings_dummies.to_csv('../data/valuation/listings_dummies_filtered.csv', index=False, compression='gzip')



### MODEL DATA ###

# import data
import pandas as pd
listings = pd.read_csv('../data/valuation/listings_make_filtered.csv', compression='gzip')
listings_dummies = pd.read_csv('../data/valuation/listings_dummies_filtered.csv', compression='gzip')
columns = listings_dummies.columns

# encode each make of interest with number
make_counts = listings['Make'].value_counts()
# makes_of_interest = make_counts[:3].index
# makes_of_interest = ['Ford', 'Chevrolet', 'Honda', 'Toyota', 'Mercedes-Benz', 'BMW', 'Volkswagen', 'Porsche']
makes_of_interest = ['Ford', 'Mercedes-Benz']
make_number = pd.DataFrame({'Make': makes_of_interest, 
                            'Make code': list(range(len(makes_of_interest)))})

# merge make numbers with selected data
listings_make_numbers = make_number.merge(listings, on='Make', how='left')


# filter out oldest models
listings_dummies_filtered = listings_dummies[listings_dummies['Age'] < 25]
listings_make_filtered = listings_make_numbers[listings_make_numbers['Age'] < 25]

### set features (independent) and labels (dependent)

# # full set of model dummies
# X = listings_dummies_filtered.drop(columns='Price')
# y = listings_dummies_filtered['Price']

# makes labels only
X = listings_make_filtered.drop(columns=['Make', 'Make code'])
y = listings_make_filtered['Make code']


# ### STATSMODELS ###

# import sys
# sys.path.insert(0, "./ancillary functions/")

# import statsmodels.api as sm
# from helper_functions import SMWrapper
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression


# # simple fit of entire data set
# model = sm.OLS(y, X)
# results_train = model.fit()
# summary_train = results_train.summary()
# summary_train_text = summary_train.as_text()
# pvalues_train = results_train.pvalues


# # cross-validation
# print(cross_val_score(SMWrapper(sm.OLS), X, y, cv=5, scoring='r2'))
# print(cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2'))


# split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # fit data - Statsmodels
# model = sm.OLS(y_train, X_train)

# # get training set summary
# results_train = model.fit()
# summary_train = results_train.summary()
# summary_train_text = summary_train.as_text()
# pvalues_train = results_train.pvalues

# # predict test set
# y_pred = results_train.predict(X_test)






### LDA - component axes that maximize class separation - supervised, considers dependent variable ### 

# apply linear discriminant analysis (LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# fit logistic regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver='lbfgs')
classifier.fit(X_train, y_train)

# predict test set results
y_pred = classifier.predict(X_test)

# create confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)


# visualize LDA results
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x_set, y_set = X, y
x1, x2 = np.meshgrid(np.arange(start = x_set.iloc[:, 0].min() - 1, 
                               stop = x_set.iloc[:, 0].max() + 1, step = 1000),
                     np.arange(start = x_set.iloc[:, 1].min() - 1, 
                               stop = x_set.iloc[:, 1].max() + 1, step = 1000))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j,
                edgecolors = 'black', s = 10)
plt.title('Logistic regression - training set')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
#plt.xlabel('Dimension 1 (%s%%)' % str(int(explained_variance[0]*100)))
#plt.ylabel('Dimension 2 (%s%%)' % str(int(explained_variance[1]*100)))
# plt.savefig('../images/LDA_1.png', bbox_inches='tight', dpi = 400) 
plt.show()
















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













