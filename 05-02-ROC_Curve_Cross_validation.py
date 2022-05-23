# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:41:56 2021

@author: HRollins
"""

# ##### Receiver Operating Characteristics (ROC) and AUC
# \begin{equation}
# \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
# \end{equation}

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve

import mglearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
X, y = make_blobs(n_samples=(4000, 500), cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


mglearn.plots.plot_decision_threshold()

# In[64]:
    
svc = SVC(gamma=.05).fit(X_train, y_train)
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

#%%
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")

# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)

plt.legend(loc=4)


# In[65]:

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, label="ROC Curve SVC")
plt.plot(fpr_rf, tpr_rf, label="ROC Curve RF")

plt.xlabel("FPR")
plt.ylabel("TPR (recall)")

plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
         label="threshold zero SVC", fillstyle="none", c='k', mew=2)

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(fpr_rf[close_default_rf], tpr[close_default_rf], '^', markersize=10,
         label="threshold 0.5 RF", fillstyle="none", c='k', mew=2)

plt.legend(loc=4)

# In[66]:

from sklearn.metrics import roc_auc_score

rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))

print("AUC for Random Forest: {:.3f}".format(rf_auc))
print("AUC for SVC: {:.3f}".format(svc_auc))

# In[3]:
# ### Cross-Validation

mglearn.plots.plot_cross_validation()

# In[4]:

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression(max_iter=1000)

scores = cross_val_score(logreg, iris.data, iris.target) #k=5 for default

predict_results = cross_val_predict(logreg, iris.data, iris.target) #k=5 for default

print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))

# In[5]:

scores = cross_val_score(logreg, iris.data, iris.target, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))

# In[7]:

from sklearn.model_selection import cross_validate
res = cross_validate(logreg, iris.data, iris.target, cv=10, return_train_score=True)

res_df = pd.DataFrame(res)
print("Mean times and scores:\n", res_df)