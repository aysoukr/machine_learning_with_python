#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataframe= pd.read_csv("FuelConsumption.csv")
dataframe.shape
dataframe.head()


# In[3]:


dataframe.describe()


# In[4]:


dataframe.columns


# In[5]:


cdf= dataframe[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(10)


# In[6]:


cdf.hist()
plt.show


# In[21]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FuelConsumption")
plt.ylabel("CO2EMISSIONS")
plt.show()


# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("FuelConsumption")
# plt.ylabel("ENGINESIZE")
# plt.show()

# plt.scatter(cdf.CO2EMISSIONS, cdf.CYLINDERS, color='blue', alpha= 0.5, marker ='D')
# plt.xlabel("CO2EMISSIONS")
# plt.ylabel("CYLINDERS")
# plt.show()

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test=train_test_split (x,y, test_size=0.2 , random_state=4)
# print('train set:',x_train.shape, y_train.shape )
# 
# print('test set:',x_test.shape, y_test.shape )

# In[27]:


msk= np.random.rand(len(dataframe))<0.8
train= cdf[msk]
test= cdf[~msk]
print('the size of train:',train.shape)
print('the size of test:',test.shape)


# # Simple Regression Model
# Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the actual value y in the dataset, and the predicted value yhat using linear approximation.
# 
# Train data distribution

# In[30]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,color='blue', marker='o', alpha=0.5)
plt.xlabel('ENGINESIZE')
plt.ylabel=('CO2EMISSIONS')
plt.show()


# In[33]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# # Plot outputs

# In[39]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,color='blue', marker='o', alpha=0.5)
plt.plot(train_x, regr.coef_[0][0]* train_x+regr.intercept_[0], 'go--', linewidth=2 )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# # Evaluation
# We compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
# 
# There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:
# 
# Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
# 
# Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean Absolute Error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
# 
# Root Mean Squared Error (RMSE).
# 
# R-squared is not an error, but rather a popular metric to measure the performance of your regression model. It represents how close the data points are to the fitted regression line. The higher the R-squared value, the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

# In[40]:


from sklearn.metrics import r2_score
test_x= np.asanyarray(test[['ENGINESIZE']])
test_y= np.asanyarray(test[['CO2EMISSIONS']])
test_y_= regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# In[ ]:




