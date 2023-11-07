#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


happiness_data=pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\happiness_score_dataset.csv')


# In[3]:


happiness_data.head()


# In[4]:


happiness_data.isnull().sum()


# In[5]:


happiness_data_columns=['Country', 'Region', 'Happiness Score', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity']


# In[6]:


happiness_data=happiness_data[happiness_data_columns].copy()


# In[7]:


happiness_data.head()


# In[8]:


happy_df=happiness_data.rename({'Country':'country', 'Region':'region', 'Happiness Score':'happiness_score', 'Economy (GDP per Capita)':'GDP_per_capita', 'Family':'family', 'Health (Life Expectancy)':'health_life_expectancy', 'Freedom':'freedom', 'Trust (Government Corruption)':'government_corruption', 'Generosity':'generosity'})


# In[9]:


happy_df.head()


# In[10]:


happy_df.isnull().sum()


# In[11]:


#plot between Happiness Score and GDP
plt.title('Plot between Happiness Score and GDP')
sns.scatterplot(x= happiness_data['Happiness Score'], y=happiness_data['Economy (GDP per Capita)'], hue=happiness_data.Region)
plt.xlabel('Happiness Score')
plt.ylabel('GDP per capita')
plt.legend(loc='upper left', fontsize='6')


# In[12]:


sns.heatmap(happiness_data.corr(), annot=True)
plt.show()


# In[13]:


# there is low correlation between generosity, trust and happiness score


# In[14]:


sns.scatterplot(x=happy_df['Freedom'], y=happy_df['Happiness Score'], hue=happy_df['Region'], data=happy_df)
plt.legend(loc='upper left', fontsize='6')
plt.xlabel('Freedom')
plt.ylabel('Happiness Score')


# In[15]:


# corruption vs happiness score
sns.scatterplot(x=happy_df['Happiness Score'], y=happy_df['Trust (Government Corruption)'], hue=happy_df['Region'], data=happy_df)
plt.legend(loc='upper left', fontsize='6')
plt.xlabel('Happiness Score')
plt.ylabel('Trust (Government Corruption)')


# In[23]:


X=happy_df.drop(columns=['Happiness Score', 'Country', 'Region'], axis=1)
y=happy_df['Happiness Score']


# In[24]:


X


# In[25]:


y


# In[26]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=10)


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


X.dtypes


# In[29]:


linreg=LinearRegression()
linreg.fit(X_train, y_train)


# In[30]:


linreg.score(X_test, y_test)


# In[31]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[32]:


dec_tree_model=DecisionTreeRegressor()


# In[33]:


dec_tree_model.fit(X_train, y_train)


# In[35]:


dt_pred= dec_tree_model.predict(X_test)
print(dt_pred)


# In[37]:


dec_tree_mse=mean_squared_error(y_test, dt_pred )
dec_tree_mse


# In[38]:


dec_tree_rmse=dec_tree_mse**0.5
dec_tree_rmse


# In[40]:


ran_for=RandomForestRegressor()


# In[41]:


ran_for.fit(X_train, y_train)
ran_for_pred=ran_for.predict(X_test)
ran_for_pred


# In[42]:


ran_for_mse=mean_squared_error(y_test, ran_for_pred )
ran_for_mse


# In[43]:


ran_for_rmse=ran_for_mse**0.5
ran_for_rmse


# In[44]:


#The RMSE measures the standard deviation of the errors between predicted and actual values. A lower value is better. Comparing the RMSE for the decision tree and the random forest, we see that the random forest provides more accurate predictions compared to the decision tree.


# In[45]:


feature_importances = ran_for.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
print(importance_df.sort_values(by='Importance', ascending=False))


# In[46]:


#The Random Forest model was more effective in predicting the happiness index compared to the Decision Tree. The most influential factors affecting the happiness index are Family, GDP per capita, and healthy life expectancy. These factors make up most of the total feature importance in the random forest model. Factors such as freedom, generosity and Government corruption also play a role, but their contribution to the overall feature importance is much smaller.


# In[ ]:




