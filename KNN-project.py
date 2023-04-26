#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv("teleCust1000t.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df["custcat"].value_counts()


# In[9]:


df.hist(column='income', bins=50)


# In[7]:


df.columns


# # To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:

# In[10]:


x= df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values   #astype(float)
x[0:5]


# In[11]:


y= df[['custcat']].values
y[0:5]


# # Normalize Data
# Data Standardization gives the data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on the distance of data points:

# In[10]:


scaler = preprocessing.StandardScaler().fit(x)
x= scaler.transform(x.astype(float))
print(x[0:5])


# # Train Test Split
# Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that that the model has NOT been trained on. Doing a train and test on the same dataset will most likely have low out-of-sample accuracy, due to the likelihood of our model overfitting.
# 
# It is important that our models have a high, out-of-sample accuracy, because the purpose of any model, of course, is to make correct predictions on unknown data. So how can we improve out-of-sample accuracy? One way is to use an evaluation approach called Train/Test Split. Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set.
# 
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that has been used to train the model. It is more realistic for real world problems.

# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split (x,y, test_size=0.2 , random_state=4)
print('train set:',x_train.shape, y_train.shape )

print('test set:',x_test.shape, y_test.shape )


# # Classification
# K nearest neighbor (KNN)
# Import library
# Classifier implementing the k-nearest neighbors vote.

# In[13]:


from sklearn.neighbors import KNeighborsClassifier


# In[14]:


k= 4
neigh= KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
neigh


# In[19]:


yhat= neigh.predict(x_test)
y_hat_train= neigh.predict(x_train)
yhat[0:5]


# # Accuracy evaluation
# In multilabel classification, accuracy classification score is a function that computes subset accuracy. This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.

# In[20]:


from sklearn import metrics
print("train set accuracy:", metrics.accuracy_score(y_train, y_hat_train) )
print("test set accuracy:", metrics.accuracy_score(y_test, yhat ) )    


# # What about other K?
# K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the user. So, how can we choose right value for K? The general solution is to reserve a part of your data for testing the accuracy of the model. Then choose k =1, use the training part for modeling, and calculate the accuracy of prediction using all samples in your test set. Repeat this process, increasing the k, and see which k is the best for your model.
# 
# We can calculate the accuracy of KNN for different values of k.

# In[24]:


Ks=10
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))

for n in range(1,Ks):
    
    #train model and predict
    neigh= KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1]= metrics.accuracy_score(y_test,yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc   
std_acc


# ## Plot the model accuracy for a different number of neighbors.

# In[33]:


plt.plot(range(1,Ks),mean_acc, linestyle='dashed')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.2 ,color="red" )
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.2,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout(pad= 2.2, h_pad=6, w_pad= 2.3)
plt.show()


# In[36]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:




