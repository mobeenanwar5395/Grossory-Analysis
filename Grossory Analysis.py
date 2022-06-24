#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk


# In[11]:


data = pd.read_csv("E:/IMS Uni Data/6th  Semester/DataScience/Grossory dataset/superstore.csv")
data


# In[6]:


data.info()


# In[17]:


data.drop(['Customer_Id'], axis = 0, inplace=True)


# In[20]:


df = pd.DataFrame(data)
df


# In[21]:


data.isnull().sum()


# In[22]:


df.fillna(method ='bfill')


# In[23]:


data.isnull().sum()


# In[24]:


data.info()


# In[25]:


data.describe()


# In[26]:


data.shape


# In[27]:


data.dtypes


# In[28]:


for col in data.columns:
    print(col)


# In[29]:


data.duplicated().sum()


# In[30]:


data.nunique()


# In[31]:


data.corr()


# In[32]:


data.cov()


# In[33]:


pd.value_counts(data.values.flatten())


# In[34]:


pd.value_counts("Country_Region")


# In[36]:


pd.value_counts("Region")


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sn


# In[47]:


plt.figure(figsize=(25,8))
plt.bar('Country_Region','Continent', data=data)
plt.show()


# In[49]:


data['Country_Region'].value_counts()


# In[51]:


plt.figure(figsize=(15,15))
sns.countplot(x=data['Country_Region'])
plt.xticks(rotation=90)
plt.show()


# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[54]:


plt.figure(figsize=(15,15))
sns.countplot(x=data['Country_Region'])
plt.xticks(rotation=90)
plt.show()


# In[57]:


data['Department'].value_counts()
plt.figure(figsize=(15,15))
sns.countplot(x=data['Department'])
plt.xticks(rotation=90)
plt.show()


# In[58]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# In[59]:


sns.heatmap(data.cov(),annot=True)
plt.show()


# In[62]:


plt.figure(figsize=(20,20))
sns.barplot(x=data['Sales'],y=data['Country_Region'])
plt.show()


# In[65]:


plt.figure(figsize=(10,4))
sns.lineplot('Profit','Sales', data=data , color='y',label='Discount')
plt.legend()
plt.show()


# In[67]:


figsize=(15,10)
sns.pairplot(data,hue='Continent')
plt.show()


# In[ ]:




