#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


data = pd.read_csv('loans_full_schema.csv')
data


# In[3]:


data.info()


# ## Visualizations

# In[4]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
sns.heatmap(data.corr())
plt.title('Correlation matix')
plt.show()


# In[5]:


# data['interest_rate'].hist()
plt.figure(figsize=(10,10))
sns.histplot(data['interest_rate'])
plt.title('Histogram of interest rate')
plt.show()


# In[6]:


from sklearn.decomposition import PCA

interest_rate = data.corr()['interest_rate']
data_clean = data[interest_rate[(interest_rate>0.1) | (interest_rate<-0.1)].keys()]
data_clean.info()


# In[7]:


data_final = data_clean.drop(columns=['annual_income_joint','debt_to_income_joint']).dropna().reset_index(drop=True)
data_final


# In[8]:


X = data_final.drop(columns=['interest_rate']).values
y = data_final['interest_rate'].values

pca = PCA()
X_pca=pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print(y)
plt.scatter(X_pca[y<15,0],X_pca[y<15,1])
plt.scatter(X_pca[y>=15,0],X_pca[y>=15,1])


# In[9]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')


xa = X_pca[y<15,0]
ya = X_pca[y<15,1]
za = X_pca[y<15,2]

ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

ax.scatter(xa, ya, za)


xb = X_pca[y>=15,0]
yb = X_pca[y>=15,1]
zb = X_pca[y>=15,2]
ax.scatter(xb, yb, zb)
plt.title('PCA with 3 components')
plt.show()


# In[10]:


data_final
for i in data_final.columns:
    plt.figure(figsize=(8,8))
    sns.boxplot(x=data_final[i])
    plt.show()


# In[11]:


interest_rate = data.corr()['interest_rate']
data_clean = data[interest_rate[(interest_rate>0.1) | (interest_rate<-0.1)].keys()]
data_final_state = data_clean.drop(columns=['annual_income_joint','debt_to_income_joint']).dropna()
data_final_state['state'] = data['state'].iloc[data_final.index]
data_final_state


# In[12]:


groups = data_final_state.groupby(by='state').mean()
inter_states = groups['interest_rate']
plt.figure(figsize=(15,15))
sns.barplot(inter_states.keys(),inter_states.values)
plt.title('Interest by State')
plt.ylabel('Interest rate')
plt.show()


# ## Machine Learning models and results

# In[13]:


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[15]:


model_svr = SVR()
model_forest = RandomForestRegressor()

# Train
model_svr.fit(x_train,y_train)
model_forest.fit(x_train,y_train)

# Evaluate
y_pred_svr = model_svr.predict(x_test)
y_pred_forest = model_forest.predict(x_test)

print('Mean absolute error for SVR is: ',mean_absolute_error(y_test,y_pred_svr))
print('Mean squared error for SVR is: ',mean_squared_error(y_test,y_pred_svr))

print('Mean absolute error for Random forest is: ',mean_absolute_error(y_test,y_pred_forest))
print('Mean squared error for Random forest is: ',mean_squared_error(y_test,y_pred_forest))


# In[16]:


plt.figure(figsize=(10,10))

sns.scatterplot(np.arange(0,y_test.shape[0]),y_test-y_pred_svr)
sns.scatterplot(np.arange(0,y_test.shape[0]),y_test-y_pred_forest)
plt.xlabel('Samples')
plt.ylabel('Error from ground truth')
plt.title('Models error from ground truth')
plt.legend(['SVR','Random Forest'])

