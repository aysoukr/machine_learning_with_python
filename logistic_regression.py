#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df= pd.read_csv('heart.csv')
df.shape


# In[6]:



df.hist(column='output' , bins=50)


# In[7]:


df['output'].value_counts()


# In[8]:


df.columns


# In[11]:


X=df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh','exng', 'oldpeak', 'slp', 'caa', 'thall']]
X[0:10]


# In[14]:


y= df[['output']]
y[1:10]


# In[16]:


X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:10]


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('train set:', X_train.shape,  y_train.shape)
print ('test set:', X_test.shape,  y_test.shape)


# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR= LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
LR


# In[25]:


yhat= LR.predict(X_test)
yhat


# In[26]:


yhat_prob= LR.predict_proba(X_test)
yhat_prob


# In[28]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test,yhat,pos_label=0)


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[30]:


cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


# In[31]:


print (classification_report(y_test, yhat))


# In[ ]:




