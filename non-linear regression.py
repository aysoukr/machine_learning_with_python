#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


# In[3]:


df= pd.read_csv('china_gdp.csv')
df.head()


# In[16]:


plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
fig.savefig('test2png.png', dpi=100)


# In[19]:


X=np.arange(-5.0,5.0,0.1)
Y= 1.0/(1.0+ np.exp(-X))
plt.plot(X,Y, "b<")
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()


# In[23]:


def sigmoid(x, beta_1, beta_2):
    y=1/(1+np.exp(-beta_1*(x-beta_2)))
    return y


# In[36]:


beta_1 = 0.8
beta_2 = 2006.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')


# In[37]:


# Lets normalize our data
xdata= x_data/max(x_data)
ydata= y_data/max(y_data)


# In[38]:


from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


# In[39]:


x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


# In[ ]:




