#!/usr/bin/env python
# coding: utf-8

# In[114]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[283]:


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


# In[284]:


train_test_data = [train_df,test_df]


# In[285]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand = False)


# In[286]:


train_df['Title'].value_counts()


# In[287]:


title_mapping = {'Mr' : 0 ,'Miss' : 1, 'Mrs' : 2, "Master" : 3, "Dr" : 3,'Rev' : 3, 'Major' : 3, 'Mlle' : 3, 'Col' : 3,'Mme' : 3,'Lady' : 3,'Capt': 3,'Don' : 3,',''Ms' : 3,'Sir' : 3,'Countess' : 3, 'Jonkheer' : 3 }


# In[288]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[289]:


train_df.head()


# In[236]:


bar_chart('Title')


# In[290]:


train_df.info()


# In[291]:


train_df.drop('Name',axis=1, inplace=True)
test_df.drop('Name',axis=1, inplace=True)


# In[239]:


train_df.head()


# In[240]:


test_df


# In[292]:


sex_mapping = {"male" : 0, "female" : 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[293]:


bar_chart('Sex')


# In[ ]:


# fill missing age with median age for each title 


# In[294]:


train_df["Age"].fillna(train_df.groupby("Title")["Age"].transform("median"),inplace=True)
test_df["Age"].fillna(test_df.groupby("Title")["Age"].transform("median"),inplace=True)


# In[198]:


age_facet = sns.FacetGrid(train_df,hue = "Survived",aspect = 4)
age_facet.map(sns.kdeplot,'Age',shade = True)
age_facet.set(xlim=(0, train_df['Age'].max()))
age_facet.add_legend()

plt.show()


# In[199]:


age_facet = sns.FacetGrid(train_df,hue = "Survived",aspect = 4)
age_facet.map(sns.kdeplot,'Age',shade = True)
age_facet.set(xlim=(0, train_df['Age'].max()))
age_facet.add_legend()

plt.xlim(0,20)


# In[200]:


age_facet = sns.FacetGrid(train_df,hue = "Survived",aspect = 4)
age_facet.map(sns.kdeplot,'Age',shade = True)
age_facet.set(xlim=(0, train_df['Age'].max()))
age_facet.add_legend()

plt.xlim(20,40)


# In[201]:


age_facet = sns.FacetGrid(train_df,hue = "Survived",aspect = 4)
age_facet.map(sns.kdeplot,'Age',shade = True)
age_facet.set(xlim=(0, train_df['Age'].max()))
age_facet.add_legend()

plt.xlim(40,60)


# In[202]:


age_facet = sns.FacetGrid(train_df,hue = "Survived",aspect = 4)
age_facet.map(sns.kdeplot,'Age',shade = True)
age_facet.set(xlim=(0, train_df['Age'].max()))
age_facet.add_legend()

plt.xlim(60,80)


# In[ ]:


Binning

feature vector map:
child : 0 (0 ~ 15)
young : 1 (16 ~ 25)
adult : 2 (26 ~ 35)
mid-age : 3 (36 ~ 45) 
old : 4 (46~)


# In[295]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0,
    dataset.loc[ (dataset['Age'] > 15) & (dataset['Age'] <= 25), 'Age'] = 1,
    dataset.loc[ (dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 2,
    dataset.loc[ (dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 45, 'Age'] = 4


# In[296]:


train_df.head()


# In[246]:


bar_chart('Age')


# In[216]:


null_embaked = train_df.loc[train_df.isnull()['Embarked'],:]


# In[220]:


print(null_embaked)


# In[221]:


null_embaked2 = test_df.loc[test_df.isnull()['Embarked'],:]


# In[226]:


test_df.info()


# In[ ]:


Embaked missing value => Pclass 1


# In[253]:


train_Pclass1 = train_df[train_df['Pclass'] == 1]['Embarked'].value_counts()
train_Pclass1.plot(kind='bar',stacked=True, figsize=(10,5))


# In[256]:


test_Pclass1 = test_df[test_df['Pclass'] == 1]['Embarked'].value_counts()
test_Pclass1.plot(kind='bar',stacked=True, figsize=(10,5))


# In[297]:


train_df['Fare'].fillna(train_df.groupby('Pclass')['Fare'].transform("median"),inplace = True )
test_df['Fare'].fillna(test_df.groupby('Pclass')['Fare'].transform("median"),inplace = True )


# In[268]:


def facet_graph(dataset,feature1,feature2,aspect_num):
    facet = sns.FacetGrid(dataset,hue = feature1, aspect = aspect_num)
    facet.map(sns.kdeplot, feature2, shade = True)
    facet.set(xlim=(0, dataset[feature2].max()))
    facet.add_legend()


# In[269]:


facet_graph(train_df,'Survived','Fare',4)
plt.show()


# In[271]:


facet_graph(train_df,'Survived','Fare',4)
plt.xlim(0,20)


# In[273]:


facet_graph(train_df,'Survived','Fare',4)
plt.xlim(20,40)


# In[274]:


facet_graph(train_df,'Survived','Fare',4)
plt.xlim(40,60)


# In[275]:


facet_graph(train_df,'Survived','Fare',4)
plt.xlim(60,80)


# In[277]:


facet_graph(train_df,'Survived','Fare',4)
plt.xlim(80,100)


# In[298]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 15, 'Fare'] = 0,
    dataset.loc[ (dataset['Fare'] > 15) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[ (dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[305]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] == 4, 'Fare'] = 3


# In[299]:


train_df.head()


# In[306]:


train_Emarked = train_df[train_df['Embarked'] == 'S']['Fare'].value_counts()


# In[307]:


train_Emarked


# In[308]:


train_Emarked2 = train_df[train_df['Embarked'] == 'C']['Fare'].value_counts()


# In[309]:


train_Emarked2


# In[310]:


test_Emarked = test_df[test_df['Embarked'] == 'S']['Fare'].value_counts()
test_Emarked2 = test_df[test_df['Embarked'] == 'C']['Fare'].value_counts()


# In[312]:


test_Emarked


# In[313]:


test_Emarked2


# In[314]:


train_df['Embarked'].fillna("S",inplace = True )
test_df['Embarked'].fillna("S",inplace = True )


# In[321]:


train_df.info()


# In[320]:


train_df['Title'].fillna(3,inplace = True )


# In[324]:


Embarked_mapping = {"S" : 0, "Q" : 1, "C" : 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(Embarked_mapping)


# In[325]:


train_df.head()


# In[328]:


train_df['Cabin'].value_counts()


# In[329]:


train_df['Cabin'].unique()


# In[406]:


for dataset in train_test_data:
    dataset['Cabin_T'] = dataset['Cabin'].str.replace(' ',"")


# In[407]:


train_df['Cabin_T'].unique()


# In[408]:


for dataset in train_test_data:
    dataset['Cabin_T'] = dataset['Cabin_T'].str.replace('[0-9]+',"")


# In[386]:


train_df['Cabin_T'].unique()


# In[394]:


train_df['Cabin_T'].value_counts()


# In[414]:


test_df['Cabin_T'].value_counts()


# In[390]:


bar_chart('Cabin_T')


# In[ ]:


죽은 사람의 객실정보가 부족.


# In[409]:


for dataset in train_test_data:
    dataset['Cabin_T'] = dataset['Cabin_T'].str.extract('([A-Z])',expand = False)


# In[395]:


bar_chart('Cabin_T')


# In[396]:


Pclass1 = train_df[train_df['Pclass'] == 1]['Cabin_T'].value_counts()
Pclass2 = train_df[train_df['Pclass'] == 2]['Cabin_T'].value_counts()
Pclass3 = train_df[train_df['Pclass'] == 3]['Cabin_T'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index =['Pclass1','Pclass2','Pclass3']
df.plot(kind = 'bar', stacked = True, figsize = (10,5))


# In[397]:


Pclass1 = test_df[test_df['Pclass'] == 1]['Cabin_T'].value_counts()
Pclass2 = test_df[test_df['Pclass'] == 2]['Cabin_T'].value_counts()
Pclass3 = test_df[test_df['Pclass'] == 3]['Cabin_T'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index =['Pclass1','Pclass2','Pclass3']
df.plot(kind = 'bar', stacked = True, figsize = (10,5))


# In[ ]:


Pclass1 = A,B,C,D,E
Pclass2 = D,E,F
Pclass3 = E,F,G

E의 경우 전 Pclass 이용
C의 경우 Pclass1 만 이용한 정보 남아있음
Pclass1의 cabin 넘버가 가장 많이 남아있음.


# In[410]:


Fare0 = train_df[train_df['Fare'] == 0]['Cabin_T'].value_counts()
Fare1 = train_df[train_df['Fare'] == 1]['Cabin_T'].value_counts()
Fare2 = train_df[train_df['Fare'] == 2]['Cabin_T'].value_counts()
Fare3 = train_df[train_df['Fare'] == 3]['Cabin_T'].value_counts()
df = pd.DataFrame([Fare0,Fare1,Fare2,Fare3])
df.index =['Fare0','Fare1','Fare2','Fare3']
df.plot(kind = 'bar', stacked = True, figsize = (10,5))


# In[415]:


Cabin_mapping = {'A' : 0 ,'B' : 1, 'C' : 2, "D" : 3, "E" : 4,'F' : 5, 'G' : 6, 'T' : 2}
for dataset in train_test_data:
    dataset['Cabin_T'] = dataset['Cabin_T'].map(Cabin_mapping)


# In[ ]:





# In[416]:


train_df['Cabin_T'].fillna(train_df.groupby('Fare')['Cabin_T'].transform("median"),inplace = True )
test_df['Cabin_T'].fillna(test_df.groupby('Fare')['Cabin_T'].transform("median"),inplace = True )


# In[417]:


train_df['Cabin_T'].value_counts()


# In[418]:


test_df['Cabin_T'].value_counts()


# In[419]:


test_df.info()

