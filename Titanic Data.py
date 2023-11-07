#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[38]:


titanic_data=pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\titanic_train.csv')


# In[39]:


titanic_data.head()


# In[40]:


titanic_data.shape


# In[41]:


titanic_data.info()


# In[42]:


titanic_data.isnull().sum()


# # Handling the missing values

# In[43]:


#drop the cabin column from the dataset
titanic_data=titanic_data.drop(columns='Cabin', axis=1)


# In[44]:


#replacing the missing values in age column with the mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[45]:


titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[46]:


titanic_data.info()


# In[47]:


titanic_data.isnull().sum()


# # Data Analysis

# In[48]:


titanic_data.describe()


# In[49]:


#finding number of people survided
titanic_data['Survived'].value_counts()


# In[50]:


#Making a count plot for survived 
sns.countplot('Survived', data=titanic_data)
plt.show()


# In[51]:


sns.countplot('Sex', data=titanic_data)
plt.show()


# In[52]:


sns.countplot('Sex', hue='Survived', data=titanic_data)
plt.show()


# In[53]:


sns.countplot('Pclass', data=titanic_data)
plt.show()


# In[54]:


sns.countplot('Pclass', hue='Survived', data=titanic_data)
plt.show()


# In[55]:


#Encoding the categorical columns
titanic_data['Sex'].value_counts()


# In[56]:


titanic_data['Embarked'].value_counts()


# In[57]:


titanic_data['Age'].plot.hist()


# In[58]:


titanic_data['Fare'].plot.hist(bins=20, figsize=(10,5))


# In[59]:


sns.countplot(x='SibSp', data=titanic_data)


# In[60]:


titanic_data['Parch'].plot.hist()


# In[61]:


sns.countplot(x='Parch', data=titanic_data)


# In[22]:


#Converting categorical columns 
titanic_data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)


# In[23]:


titanic_data.head()


# In[24]:


sns.heatmap(titanic_data.corr(), annot=True)
plt.show()


# In[62]:


sns.boxplot(x='Pclass', y='Age', data=titanic_data)


# In[63]:


#We can observe that older people are travelling by class 1 & 2 


# # Seperating features and target
# 

# In[25]:


X=titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y=titanic_data['Survived']


# In[26]:


print(X)


# In[27]:


print(y)


# # Splitting the data into training and testing data

# In[28]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=10)


# In[29]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# # Model Training

# # Logistic Regression

# In[30]:


logreg=LogisticRegression()


# In[31]:


#training the logreg model with training data
logreg.fit(X_train, y_train)


# In[32]:


#Model Evaluation

X_train_prediction=logreg.predict(X_train)


# In[33]:


print(X_train_prediction)


# In[34]:


training_data_accuracy=accuracy_score(y_train, X_train_prediction)
print(training_data_accuracy)


# In[35]:


X_test_prediction=logreg.predict(X_test)
print(X_test_prediction)


# In[36]:


test_data_accuracy=accuracy_score(y_test, X_test_prediction)
print(test_data_accuracy)


# In[ ]:




