#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv("FuelConsumption.csv")
df.head()


# In[4]:


dcf= df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
dcf.head(9)


# In[13]:


plt.scatter(dcf.ENGINESIZE, dcf.CO2EMISSIONS, color='blue')
plt.xlabel("ENGINE SIZE")
plt.ylabel('CO2 EMISSIONS')
plt.show()
dcf.max()


# In[12]:


msk=np.random.rand(len(dcf))<0.8
train= dcf[msk]
test=dcf[~msk]
print(train)
print(test)


# # PolynomialFeatures() function:
# in Scikit-learn library, drives a new feature sets from the original feature set. That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, lets say the original feature set has only one feature, ENGINESIZE. Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:

# In[16]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x=np.asanyarray(train[[ 'ENGINESIZE']])
train_y=np.asanyarray(train[[ 'CO2EMISSIONS']])
test_x=np.asanyarray(test[[ 'ENGINESIZE']])
test_y=np.asanyarray(test[[ 'CO2EMISSIONS']])
poly=PolynomialFeatures(degree=3)
train_x_poly=poly.fit_transform(train_x)
train_x_poly


# In[17]:


clf=linear_model.LinearRegression()
train_y_=clf.fit(train_x_poly, train_y)
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)


# In[20]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX=np.arange(0.0, 10.0, 0.1)
yy= clf.intercept_[0]+clf.coef_[0][1]*XX+clf.coef_[0][2]*np.power(XX,2)+clf.coef_[0][3]*np.power(XX,3)
plt.plot(XX,yy, "-r")
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')


# In[21]:


from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )


# In[ ]:




