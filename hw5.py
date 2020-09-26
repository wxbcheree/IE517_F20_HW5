#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:59:21 2020

@author: xuebinwang
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import math
from sklearn.svm import SVR


df = pd.read_csv("hw5_treasury yield curve data.csv")
del df['Date']
X, y = df.iloc[:, :-1].values, df.loc[:, ['Adj_Close']]
df.columns = ['SVENF1','SVENF2','SVENF3','SVENF4','SVENF5',
                   'SVENF6','SVENF7','SVENF8','SVENF9','SVENF10',
                   'SVENF11','SVENF12','SVENF13','SVENF14','SVENF15',
                   'SVENF16','SVENF17','SVENF18','SVENF19','SVENF20',
                   'SVENF21','SVENF22','SVENF23','SVENF24','SVENF25',
                   'SVENF26','SVENF27','SVENF28','SVENF29','SVENF30','Adj_Close']


#Scatter plot
cols = ['SVENF5','SVENF15', 'SVENF25','SVENF30','Adj_Close']
sns.pairplot(df[cols])
plt.show()

#heatmap
cols1 = ['SVENF1','SVENF2', 'SVENF3','SVENF4','SVENF5',
         'SVENF6','SVENF7', 'SVENF8','SVENF9','SVENF10']
cm = np.corrcoef(df[cols1].values.T)
hm = sns.heatmap(cm, annot = True, yticklabels=  df[cols1].columns, 
                 xticklabels=df[cols1].columns, annot_kws={"size":8})
plt.show()


cols2 = ['SVENF21','SVENF22', 'SVENF23','SVENF24','SVENF25',
         'SVENF26','SVENF27', 'SVENF28','SVENF29','SVENF30','Adj_Close']
cm = np.corrcoef(df[cols2].values.T)
hm = sns.heatmap(cm, annot = True, yticklabels=  df[cols2].columns, 
                 xticklabels=df[cols2].columns, annot_kws={"size":8})
plt.show()

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                     random_state=42)

#standardlize 
sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# PCA
pca_none = PCA()
X_train_pca_none = pca_none.fit_transform(X_train_std)
X_test_pca_none = pca_none.transform(X_test_std)
print(pca_none.explained_variance_ratio_)

pca_3 = PCA(n_components=3)
X_train_pca_3 = pca_3.fit_transform(X_train_std)
X_test_pca_3 = pca_3.transform(X_test_std)
print(pca_3.explained_variance_ratio_)
print(pca_3.explained_variance_ratio_[0]+pca_3.explained_variance_ratio_[1]+
      pca_3.explained_variance_ratio_[2])

# linear model to all feature
lr_all = LinearRegression()
lr_all.fit(X_train_std, y_train)
y_train_predlr_all = lr_all.predict(X_train_std)
y_test_predlr_all = lr_all.predict(X_test_std)
print(lr_all.coef_)
print('RMSE train: %.3f, test: %.3f' % (
        math.sqrt(mean_squared_error(y_train, y_train_predlr_all)),
        mean_squared_error(y_test, y_test_predlr_all)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_predlr_all),
        r2_score(y_test, y_test_predlr_all)))

#linear model with PCA
lr_3 = LinearRegression()
lr_3.fit(X_train_pca_3, y_train)
y_train_predlr_3 = lr_3.predict(X_train_pca_3)
y_test_predlr_3 = lr_3.predict(X_test_pca_3)
print(lr_3.coef_)
print('RMSE train: %.3f, test: %.3f' % (
        math.sqrt(mean_squared_error(y_train, y_train_predlr_3)),
        mean_squared_error(y_test, y_test_predlr_3)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_predlr_3),
        r2_score(y_test, y_test_predlr_3)))


#SVM regressor to all feature
svm_all = SVR(kernel='linear')
svm_all.fit(X_train_std, y_train)
y_train_pred_svm_all = svm_all.predict(X_train_std)


y_test_pred_svm_all = svm_all.predict(X_test_std)

print('RMSE train: %.3f, test: %.3f' % (
        math.sqrt(mean_squared_error(y_train, y_train_pred_svm_all)),
        mean_squared_error(y_test, y_test_pred_svm_all)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred_svm_all),
        r2_score(y_test, y_test_pred_svm_all)))


# SVM regressor to 3 principal
svm_3 = SVR(kernel='linear')
svm_3.fit(X_train_pca_3, y_train)
y_train_pred_svm_3 = svm_3.predict(X_train_pca_3)

y_test_pred_svm_3 = svm_3.predict(X_test_pca_3)

print('RMSE train: %.3f, test: %.3f' % (
        math.sqrt(mean_squared_error(y_train, y_train_pred_svm_3)),
        mean_squared_error(y_test, y_test_pred_svm_3)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred_svm_3),
        r2_score(y_test, y_test_pred_svm_3)))


print("My name is Xuebin Wang")
print("My NetID is: xuebinw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")







    



