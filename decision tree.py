#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[2]:


my_data = pd.read_csv("drug200.csv")
my_data[0:5]


# In[3]:


my_data.size


# In[4]:


my_data.columns


# In[10]:


X=my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# In[11]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]



# In[12]:


y= my_data['Drug']
y[0:5]


# In[14]:


from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset= train_test_split(X, y, test_size=0.3, random_state=3)


# In[17]:


drugTree= DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree #The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain
#The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples


# In[18]:


drugTree.fit(X_trainset, y_trainset)


# In[20]:


predTree= drugTree.predict(X_testset)


# In[21]:


print (predTree [0:5])
print (y_testset [0:5])


# In[23]:


from  io import StringIO
import matplotlib.pyplot as plt
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# In[ ]:




