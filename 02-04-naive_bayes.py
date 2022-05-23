# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:57:20 2021

@author: HRollins
"""
import numpy as np
from sklearn.model_selection import train_test_split

# In[54]: Naive Bayes Classifiers

from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp   =['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play   =['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','No','No','Yes','No']

le_w = preprocessing.LabelEncoder()
le_t = preprocessing.LabelEncoder()
le_p = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded=le_w.fit_transform(weather)
temp_encoded=le_t.fit_transform(temp)

X = np.stack((weather_encoded, temp_encoded)).T
y = le_p.fit_transform(play)
#y = np.zeros(14,dtype=int).T

clf = MultinomialNB()
clf.fit(X, y)

#Check a test sample
w = le_w.transform(['Overcast','Rainy','Overcast','Sunny'])
t = le_t.transform(['Hot','Mild','Cool','Cool'])

X_test = np.array([w,t]).T

playOutput = clf.predict(X_test)
print(le_p.inverse_transform(playOutput))


# In[55] Gaussian Naive Bayes

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)

print("Training set accuracy: {:.2f}".format(gnb.score(X_train, y_train)))
print("Test set accuracy: {:.2f}".format(gnb.score(X_test, y_test)))




