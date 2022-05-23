# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:43:48 2021

@author: HRollins
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Classification Example

from sklearn.datasets import load_digits
digits_dataset = load_digits()

# In[21]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    digits_dataset['data'], digits_dataset['target'], random_state=0)

# In[25]: Training the Classification Model

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# %% Testing 
y_pred = knn.predict(X_test)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Regression Example

dataset = pd.read_csv('./data/student_scores.csv')
print(dataset.head())

#%% Plot the dataset

dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

#%%
#X = dataset['Hours'].to_numpy()
#y = dataset['Scores'].to_numpy()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#%%
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

#%%

dataset.plot(kind='scatter',y='Scores',x='Hours');
plt.plot(X_test,y_pred,linewidth=3,color='red');


