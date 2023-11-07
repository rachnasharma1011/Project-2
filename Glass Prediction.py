#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# In[3]:


column_names=['Id no', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']


# In[4]:


glass_data=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset3/main/glass.csv', names=column_names)


# In[5]:


glass_data.head()


# In[6]:


glass_data.shape


# In[7]:


glass_data.info()


# In[8]:


glass_data.isnull().sum()


# In[9]:


glass_data.dtypes


# In[10]:


glass_data.describe()


# # Exploratory data Analysis and Data visualisation

# In[11]:


glass_data['Type'].value_counts()


# In[12]:


plt.figure(figsize=(10,8))
sns.countplot(glass_data['Type'])
plt.title('Distribution of Classses')


# In[13]:


#Finding variance in data
glass_data.var()


# In[14]:


#Finding skewness in data and feature transformation to remove skewness
glass_data.agg(['skew', 'kurtosis']).transpose()


# In[15]:


sns.distplot(glass_data['Na'])


# In[16]:


sns.distplot(glass_data['Mg'])


# In[17]:


sns.distplot(glass_data['Al'])


# In[18]:


sns.distplot(glass_data['Si'])


# In[19]:


sns.distplot(glass_data['K'])


# In[20]:


sns.distplot(glass_data['Ca'])


# In[21]:


sns.distplot(glass_data['Ba'])


# In[22]:


sns.distplot(glass_data['Fe'])


# In[23]:


glass_data['RI']=np.sqrt(glass_data['RI'])
glass_data['K']=np.sqrt(glass_data['K'])
glass_data['Ca']=np.sqrt(glass_data['Ca'])
glass_data['Ba']=np.sqrt(glass_data['Ba'])
glass_data['Fe']=np.sqrt(glass_data['Fe'])


# In[24]:


sns.distplot(glass_data['RI'])


# In[25]:


sns.distplot(glass_data['K'])


# In[26]:


sns.distplot(glass_data['Ca'])


# In[27]:


sns.distplot(glass_data['Ba'])


# In[28]:


sns.distplot(glass_data['Fe'])


# In[29]:


plt.subplots(figsize=(20,15))
plt.subplot(3, 3, 1)
sns.boxplot(x='Type', y='RI', data=glass_data)
plt.subplot(3, 3, 2)
sns.boxplot(x='Type', y='Na', data=glass_data)
plt.subplot(3, 3, 3)
sns.boxplot(x='Type', y='Mg', data=glass_data)
plt.subplot(3, 3, 4)
sns.boxplot(x='Type', y='Al', data=glass_data)
plt.subplot(3, 3, 5)
sns.boxplot(x='Type', y='Si', data=glass_data)
plt.subplot(3, 3, 6)
sns.boxplot(x='Type', y='K', data=glass_data)
plt.subplot(3, 3, 7)
sns.boxplot(x='Type', y='Ca', data=glass_data)
plt.subplot(3, 3, 8)
sns.boxplot(x='Type', y='Ba', data=glass_data)
plt.subplot(3, 3, 9)
sns.boxplot(x='Type', y='Fe', data=glass_data)


# In[30]:


glass_data=glass_data.drop(columns='Id no', axis=1)
glass_data.head()


# In[31]:


glass_data.shape


# In[32]:


#Seperating features and labels
features=glass_data.drop(columns='Type', axis=1)
features


# In[33]:


label=glass_data['Type']
label


# In[34]:


column=glass_data.columns
column


# In[35]:


#Multivariate plotting
sns.pairplot(glass_data)
plt.show()


# In[36]:


sns.heatmap(glass_data.corr(), annot=True)


# # Model Building

# In[37]:


X_train, X_test, y_train, y_test=train_test_split(features, label, test_size=0.2, random_state=10)


# In[38]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[39]:


from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix, precision_score, recall_score


# In[40]:


#Logistic Regression
lr=LogisticRegression()
lr.fit(X_train, y_train)
lr_pred=lr.predict(X_test)
lr_acc=accuracy_score(y_test, lr_pred)*100
print('LR accuracy score is', lr_acc)
print(classification_report(y_test, lr_pred))


# In[41]:


cm_lr=confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[42]:


#Random Forest Classifier
rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred=rfc.predict(X_test)
rfc_acc=accuracy_score(y_test, rfc_pred)*100
print('RFC accuracy score is', rfc_acc)
print(classification_report(y_test, rfc_pred))


# In[43]:


cm_rfc=confusion_matrix(y_test, rfc_pred)
sns.heatmap(cm_rfc, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[44]:


#Decision Tree Classifier
dt=DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred=dt.predict(X_test)
dt_acc=accuracy_score(y_test, dt_pred)*100
print('DT accuracy score is', dt_acc)
print(classification_report(y_test, dt_pred))


# In[45]:


cm_dt=confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[46]:


#AdaBoost Classifier
adb=AdaBoostClassifier()
adb.fit(X_train, y_train)
adb_pred=adb.predict(X_test)
adb_acc=accuracy_score(y_test, adb_pred)*100
print('AdaBoost accuracy score is', adb_acc)
print(classification_report(y_test, adb_pred))


# In[47]:


cm_adb=confusion_matrix(y_test, adb_pred)
sns.heatmap(cm_adb, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[48]:


#Gaussian Naive Bayes Classifier
gnb=GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred=gnb.predict(X_test)
gnb_acc=accuracy_score(y_test, gnb_pred)*100
print('GaussianNB accuracy score is', gnb_acc)
print(classification_report(y_test, gnb_pred))


# In[49]:


cm_gnb=confusion_matrix(y_test, gnb_pred)
sns.heatmap(cm_gnb, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[50]:


#Gradient Boosting Classifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_pred=gnb.predict(X_test)
gbc_acc=accuracy_score(y_test, gbc_pred)*100
print('Gradient Boosting accuracy score is', gbc_acc)
print(classification_report(y_test, gbc_pred))


# In[51]:


cm_gbc=confusion_matrix(y_test, gbc_pred)
sns.heatmap(cm_gbc, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# # Model Comparison

# In[52]:


model_df=pd.DataFrame({'Models':['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier', 'AdaBoost Classifier', 'Gaussian Naive Bayes Classifier', 'Gradient Boosting Classifier'], 'Accuracy Score' : [lr_acc, rfc_acc, dt_acc, adb_acc, gnb_acc, gbc_acc]})
round(model_df.sort_values(by='Accuracy Score', ascending=False), 3)


# In[ ]:




