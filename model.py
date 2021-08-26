#!/usr/bin/env python
# coding: utf-8



# In[2]:


##Importing Required Libraries

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from sklearn.metrics import roc_auc_score,roc_curve,auc
import pickle


# In[3]:


import pandas as pd
data=pd.read_csv('final_dataset.csv')
print(data.head())
# having a glance at data 


# In[4]:


# Removing unnecessary column named "Unnamed: 0"
data.drop('Unnamed: 0',axis=1,inplace=True)
data.head()
#checking whether it is droped or not!


# In[5]:


data.describe()


# In[6]:


# Checkinng shape of dataset
data.shape


# In[7]:


data.info


# In[8]:


# Plotting graph of Top100
sns.countplot(x="Top100",data=data,palette="Set3")


# In[9]:


data['Top100'].value_counts()


# In[10]:


# Splitting the data into X and y
X=data.drop("Top100",axis=1)    # excluding output column
y=data["Top100"]


# In[11]:


X.head()


# In[12]:


X.shape


# In[13]:


y.head()


# In[14]:
    
y.shape

# In[16]:

# Applying SMOTE: to balance the dataset

from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy="minority")
X_sm,y_sm=smote.fit_resample(X,y)
X_sm.shape,y_sm.shape 
# In[17]:


# Checking the shape again after applying smote
X_sm.shape,y_sm.shape


# In[18]:
    
data.columns
# In[19]:

column=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
       'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Genre']
X_balance=pd.DataFrame(X_sm,columns=column)
X_balance.head()


# In[20]:


X_balance.shape


# In[21]:


# MinMax scaler to scale our data , since we can see that there can be units difference
scaling=MinMaxScaler()
X_scaled=scaling.fit_transform(X_sm)


# In[22]:


column=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
       'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Genre']
X_final=pd.DataFrame(X_scaled,columns=column)
X_final.head(2)


# In[23]:


X_final.shape


# In[24]:


# Hold-out validation

X_train, X_test, y_train, y_test = train_test_split(X_final, y_sm, train_size = 0.8, test_size=0.2, random_state=1)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


# In[25]:


#plotting graph again to check for balanced datset
ax = sns.countplot(x = y_train, palette = "Set3")


# ## **KNN** 

# In[26]:


#create KNN object 
knn = KNeighborsClassifier(n_neighbors=10)


# In[27]:


knn.fit(X_train,y_train)


# In[28]:


#Predict test data set.
y_pred = knn.predict(X_test)


# In[29]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[30]:


#Checking performance our model with classification report.
print(classification_report(y_test, y_pred))


# In[31]:


print(accuracy_score(y_test,y_pred)*100)


# In[32]:


#Checking performance our model with ROC Score.
roc_auc_score(y_test, y_pred)


# In[33]:


from sklearn import metrics
fpr,tpr,_= metrics.roc_curve(y_test,y_pred)
auc=metrics.roc_auc_score(y_test,y_pred)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label="validation,auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.title('AUC Curve for K-NN')
plt.legend(loc=4)
plt.show()


# ##**Hyperparameter Tuning for K-NN**
# In[34]:


from sklearn.model_selection import GridSearchCV
grid_params = { 'n_neighbors' : [5,10,15,18,20,25,30],
                'weights' : ['uniform','distance'],
                'metric' : ['minkowski','euclidean','manhattan']}


# In[35]:


gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 2, cv=10, n_jobs = -1)


# In[36]:


g_res = gs.fit(X_train, y_train)


# In[37]:
    
print(gs.best_estimator_,'\n')
print(gs.best_params_,'\n')
y_gs=gs.predict(X_test)
print(accuracy_score(y_test,y_gs)*100)

# In[38]:

print(classification_report(y_test, y_gs))

# In[39]:

#Checking performance our model with ROC Score.
roc_auc_score(y_test, y_gs)


# In[40]:


from sklearn import metrics
fpr,tpr,_= metrics.roc_curve(y_test,y_pred)
auc=metrics.roc_auc_score(y_test,y_gs)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label="validation,auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.title('AUC Curve for K-NN')
plt.legend(loc=4)
plt.show()


# ## **LDA**  - Linear Discriminant Analysis

# In[41]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
lda_pred = lda.predict(X_test)


# In[42]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, lda_pred)


# In[43]:


#Checking performance our model with classification report.
print(classification_report(y_test, lda_pred))


# In[44]:


#Accuracy score
print(accuracy_score(y_test,lda_pred)*100)


# In[45]:


#Checking performance our model with ROC Score.
roc_auc_score(y_test, lda_pred)


# In[46]:


from sklearn import metrics
fpr,tpr,_= metrics.roc_curve(y_test,lda_pred)
auc=metrics.roc_auc_score(y_test,lda_pred)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label="validation,auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.title("AUC Curve for LDA")
plt.legend(loc=4)
plt.show()


# In[50]:


pickle.dump(gs.best_estimator_, open('model.pkl','wb'))


# In[52]:


model=pickle.load(open('model.pkl','rb'))


# In[53]:

print(model.predict([[0.8,0.7,0.8,0.9,0.7,0.9,0.9,0.9,0.9,8]]))
