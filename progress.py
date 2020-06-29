#!/usr/bin/env python
# coding: utf-8

# In[114]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[135]:


train_df = pd.read_csv("C:/Users/kbs03/Desktop/PythonWorkspace/train.csv")
test_df = pd.read_csv("C:/Users/kbs03/Desktop/PythonWorkspace/test.csv")


# In[97]:


train_df


# In[98]:


train_df.info()


# In[99]:


train_df.describe(include ="all")


# In[100]:


train_df.describe(include ="O")


# In[109]:


train_df[['Survived','Pclass']].groupby(['Pclass'],as_index = False ).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# 클래스가 높을 수록 생존률이 높음


# In[110]:


train_df[['Survived','Sex']].groupby(['Sex'],as_index = False ).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# 여성의 생존률이 압도적으로 높음


# In[111]:


train_df[['Survived','SibSp']].groupby(['SibSp'],as_index = False ).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# 1,0,2 > 3,4,5,6


# In[112]:


train_df[['Survived','Parch']].groupby(['Parch'],as_index = False ).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


# 3,1,2,0 vs 5,4,6


# In[118]:


graph_Age = sns.FacetGrid(train_df,col='Survived')
graph_Age.map(plt.hist,'Age',bins = 20)


# In[122]:


graph_Pclass = sns.FacetGrid(train_df,col='Survived',row='Pclass',size=2.2,aspect=1.6)
graph_Pclass.map(plt.hist,'Age',bins = 20)
graph_Pclass.add_legend();


# In[127]:


def bar_chart(feature):
    survived = train_df[train_df['Survived'] == 1][feature].value_counts()
    dead = train_df[train_df['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index =['Survived','Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))
    


# In[128]:


bar_chart('Sex')


# In[129]:


bar_chart('Pclass')


# In[131]:


bar_chart('SibSp')


# In[132]:


bar_chart('Parch')


# In[134]:


bar_chart('Embarked')


# In[136]:


train_test_data = [train_df,test_df]


# In[165]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand = False)


# In[166]:


train_df['Title'].value_counts()


# In[168]:


title_mapping = {'Mr' : 0 ,'Miss' : 1, 'Mrs' : 2, "Master" : 3, "Dr" : 3,'Rev' : 3, 'Major' : 3, 'Mlle' : 3, 'Col' : 3,'Mme' : 3,'Lady' : 3,'Capt': 3,'Don' : 3,'Ms' : 3,'Sir' : 3,'Countess' : 3, 'Jonkheer' : 3 }


# In[170]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[172]:


train_df.head()


# In[173]:


bar_chart('Title')


# In[ ]:




