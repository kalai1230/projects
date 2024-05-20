#!/usr/bin/env python
# coding: utf-8

# #  Diamond price prediction Analysis

# In[ ]:


Dataset-1
dataset link :https://www.kaggle.com/code/karnikakapoor/diamond-price-prediction/input


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[4]:


data=pd.read_csv("diamonds.csv")


# In[5]:


data


# In[6]:


df=data.copy()
df


# In[7]:


sns.histplot(x=df.carat)


# In[8]:


q1=df["carat"].quantile(0.25)
q3=df["carat"].quantile(0.75)
iqr=q3-q1
iqr
lower=q1 - 1.5*iqr
upper=q3 + 1.5*iqr
upper


# In[9]:


# count=((df["carat"]<lower)|(df["carat"]>upper)).sum()
# per=(count/len(df["carat"]))*100
# per
num=df.select_dtypes(include="float64")
#num.drop("price", axis=1,inplace=True)
num


# In[10]:


Q1=num.quantile(0.25)
Q3=num.quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
outliers_count=((num<lower)|(num>upper)).sum()
outliers_percentage=(outliers_count/len(num))*100
print(outliers_percentage)
print(outliers_count)


# In[11]:


#check the distribution 
plt.figure(figsize=(30,25),facecolor='orange')
plotnum=1
for i in num.columns:
    if(plotnum<7):
        ax=plt.subplot(3,4,plotnum)
        sns.distplot(num[i])
    plotnum+=1
plt.tight_layout()


# In[12]:


# carat column
q1=df.carat.quantile(0.25)
q3=df.carat.quantile(0.75)
iqr=q3-q1
iqr
mini=q1-1.5*iqr
maxi=q3+1.5*iqr
df.loc[(df["carat"]<mini) |(df["carat"]>maxi),"carat"]=df["carat"].median()


# In[13]:


#depth
q1=df.depth.quantile(0.25)
q3=df.depth.quantile(0.75)
iqr=q3-q1
iqr
mini=q1-1.5*iqr
maxi=q3+1.5*iqr
df.loc[(df["depth"]<mini) |(df["depth"]>maxi),"depth"]=df["depth"].mean()
df.loc[(df["depth"]<mini) |(df["depth"]>maxi),"depth"]


# In[14]:


df.loc[(df["depth"]<mini) |(df["depth"]>maxi),"depth"]


# In[15]:


#table
mean = np.mean(df.table)

std = np.std(df.table)

mini = mean-3*std
maxi = mean+3*std
df.loc[(df["table"]<mini)|(df["table"]>maxi),"table"]=df["table"].mean()


# In[16]:


df.loc[(df["table"]<mini)|(df["table"]>maxi),"table"]


# In[17]:


#column x


# In[18]:


q1=df.x.quantile(0.25)
q3=df.x.quantile(0.75)
iqr=q3-q1
iqr
mini=q1-1.5*iqr
maxi=q3+1.5*iqr
df.loc[(df["x"]<mini) |(df["x"]>maxi),"x"]=df["x"].median()


# In[19]:


df.loc[(df["x"]<mini) |(df["x"]>maxi),"x"]


# In[20]:


sns.boxplot(data=df, x=df.x)


# In[21]:


df.drop("Unnamed: 0", axis=1,inplace=True)
df


# In[22]:


# plt.figure(figsize=(10,8))
# corr=(data.select_dtypes(exclude="object").corr())
# corr
# sns.heatmap(corr, annot=True)


# In[23]:


corr=df.select_dtypes(exclude="object").corr()
sns.heatmap(corr,annot=True)


# In[24]:


df.drop("z", axis=1, inplace=True)
df


# In[25]:


#label encoding 
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.select_dtypes(include="object"):
    df[i]=l.fit_transform(df[i])


# In[26]:


df


# In[27]:


#train and split
X=df.drop("price" ,axis=1)
y=df.price


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(X,y, random_state=13, test_size=0.2)


# In[29]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[30]:


#scaling
from sklearn.preprocessing import MinMaxScaler
s=MinMaxScaler()


# In[31]:


x_train[['carat', 'depth','cut','color','clarity','table', 'x','y']]=s.fit_transform(x_train[['carat', 'depth','cut','color','clarity','table', 'x','y']])
x_train


# In[32]:


x_test[['carat','depth','cut','color','clarity','table', 'x','y']]=s.transform(x_test[['carat', 'depth', 'cut','color','clarity','table', 'x','y']])
x_test


# In[33]:


# model training
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[34]:


y_pred=model.predict(x_test)
y_pred


# In[35]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse=mean_squared_error(y_test,y_pred)
mse


# In[36]:


mae=mean_absolute_error(y_test, y_pred)
mae


# In[37]:


r2_score(y_test,y_pred)


# In[38]:


# adjusted r2 score 
adj_r2=1-(1-0.81)*(5394-1)/(5394-8-1)
adj_r2


# In[ ]:


# adj r2 score should be less than a r2 


# In[ ]:


#Ridge 


# In[118]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge(random_state=10)
parameters={'alpha':[0.01,0.001]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring="neg_root_mean_squared_error",cv=15)
ridge_regressor.fit(x_train,y_train)


# In[119]:


y_pred_r=ridge_regressor.predict(x_test)
y_pred_r


# In[120]:


r2_score(y_test,y_pred_r)


# In[121]:


#lasso 
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=0.01,random_state=13)
lasso.fit(x_train,y_train)


# In[122]:


y_pred_la=lasso.predict(x_test)
y_pred_la


# In[76]:


r2_score(y_test,y_pred_la)


# In[47]:


conclusion:
# when i tried both Min max and standard scaler nothing improvement is there 
# random state 13 transforms the accuracy to from 80 t0 81
# used ridge and apply GridsearchCV to control the overfitting


# In[1]:


pwd

