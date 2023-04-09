#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# # reading the data in

# In[9]:


df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()


# In[10]:


df.columns


# In[11]:


cdf= df[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB', 'CO2EMISSIONS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY'  ]]
cdf[1:10]


# In[12]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('ENGINE SIZE')
plt.ylabel('CO2 EMISSIONS')
plt.show()


# In[13]:


mask= np.random.rand(len(df))<0.8
train= cdf[mask]
test= cdf[~mask]


# In[15]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel('ENGINE SIZE')
plt.ylabel('CO2 EMISSIONS')
plt.show()


# In[17]:


from sklearn import linear_model
regr= linear_model.LinearRegression()
x= np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print("coefficient:", regr.coef_)
print('intercept:', regr.intercept_)


# ## As mentioned before, Coefficient and Intercept are the parameters of the fitted line. Given that it is a multiple linear regression model with 3 parameters and that the parameters are the intercept and coefficients of the hyperplane, sklearn can estimate them from our data. Scikit-learn uses plain Ordinary Least Squares method to solve this problem.
# 
# Ordinary Least Squares (OLS)
# OLS is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target dependent variable and those predicted by the linear function. In other words, it tries to minimizes the sum of squared errors (SSE) or mean squared error (MSE) between the target variable (y) and our predicted output (
# ) over all samples in the dataset.
# 
# OLS can find the best parameters using of the following methods:
# 
# Solving the model parameters analytically using closed-form equations
# Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)

# In[20]:


y_hat=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean(y_hat-y)**2)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f'%regr.score(x,y))


# In[ ]:




