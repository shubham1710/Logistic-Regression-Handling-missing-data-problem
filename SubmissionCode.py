#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
a=np.genfromtxt('train_X.csv',delimiter=',')
b=np.genfromtxt('train_Y.csv',delimiter=',')
a=np.delete(a,0,0)
col_mean = np.nanmean(a, axis = 0)
inds = np.where(np.isnan(a))
a[inds] = np.take(col_mean, inds[1])
sc = StandardScaler()
at=sc.fit_transform(a)
logreg = LogisticRegression()
logreg.fit(at,b)
y_pred=logreg.predict(at)
co=np.shape(at)[0]
y_pred.resize(co,1)
np.savetxt("predicted_test_Y.csv", y_pred, delimiter=",")


# In[ ]:




