#!/usr/bin/env python
# coding: utf-8

#  # Importing data 

# In[1]:


import numpy as np 
import pandas as pd
df_train = pd.read_csv('train.csv')

df_train.head()


# In[2]:


df_train.info()


# # Feature selection & preprocessing 
# In this section we are going to apply some preprocessing to our data so that it can fit to our classification models, we are going to proceed as following : 
# - Data cleaning
# - Spliting data
# - Text mining 

# ### Data cleaning 
# let's clean data from punctuation, and transforme words into lowercase characters 

# In[3]:


y = df_train['Label']
X = df_train['TweetText']
X= X.str.replace('[^\w\s]','')
X


# In[4]:


X = X.str.lower()
X


# In[5]:


X.isnull().sum() 


# ### Text mining & Spliting data
# in this section we are going to apply text preprocessing, tokenization, and filtering  of stopwords
# which will help us in determining the frequency of each word.

# In[6]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[7]:


ct = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())])
X_mined = ct.fit_transform(X)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_mined, y, test_size=0.22, random_state=4)


# # Model training
# in this section we are going to try different classification algorithms 

# In[9]:


Models = {'LinearSVC' : LinearSVC() , 
          'KNeighborsClassifie' : KNeighborsClassifier(),
          'SVC' : SVC(),
          'LogisticRegression' : LogisticRegression(random_state=0),
          'DesicionTree ' : DecisionTreeClassifier()
         }
result = {}


# In[10]:


for model_name, model in Models.items():

    model.fit(X_train, y_train)
    result[model_name] = model.score(X_test,y_test)
result


# # Hyper-parameter tuning
# in this section we are going to make our models better, we are going to  chose the appropriate parameters for each model, by doing this we are going to customize our models for our problem and as a result  models will become  stronger and predict better 

# In[11]:


from sklearn.model_selection import GridSearchCV 


# ##  LinearSVC()

# In[12]:


params = {'C' : [0.001, 0.01, 0.1, 1, 10]}
clf = GridSearchCV(LinearSVC(), params, cv=5)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
print(clf.best_estimator_)


# ## KNeighborsClassifier

# In[13]:


params = {'n_neighbors' : np.arange(1,10).tolist(),
         'weights': ['uniform', 'distance'],
         'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}

clf = GridSearchCV(KNeighborsClassifier(), params, cv=5)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
print(clf.best_estimator_)


# # SVC

# In[14]:


params = {'C': [1, 10], 
          'gamma': [0.001, 0.01, 1]}
clf = GridSearchCV(SVC(), params, cv=5)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
print(clf.best_estimator_)


# # LogisticRegression

# In[15]:


params = {'C':np.logspace(-3,3,7), 'penalty': ["l1","l2"]}
clf = GridSearchCV(LogisticRegression(), params, cv=5)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
print(clf.best_estimator_)


# # DecisionTreeClassifier

# In[16]:


params = { 'criterion':['gini','entropy'],
              'max_depth': np.arange(1, 15),
             }
clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
print(clf.best_estimator_)


# We have noticed that GridSearch made our models perform better, now lets use the gridsearch parameters and recreat our models 

# In[17]:


from sklearn import metrics
Models = {'LinearSVC' : LinearSVC(C=10) , 
          'KNeighborsClassifie' :KNeighborsClassifier(n_neighbors=6, weights='distance'),
          'SVC' : SVC(C=10, gamma=1),
          'LogisticRegression' :LogisticRegression(C=1000.0) ,
          'DesicionTree ' : DecisionTreeClassifier()
         }
result = {}
for model_name, model in Models.items():

    model.fit(X_train, y_train)
    yhat=model.predict(X_test)
    result[model_name] = model.score(X_test,y_test)
    print(model_name)
    print(metrics.classification_report(y_test, yhat))


# # Let's move on now to the test data 

# In[18]:


from sklearn.naive_bayes import MultinomialNB
df_test = pd.read_csv('test.csv')
Xtest = df_test['TweetText']
Xtest = Xtest.str.lower() 
Xtest = Xtest.str.replace('[^\w\s]','') 
ct= Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB(alpha=0.03)) ])
ct.fit(X, y)
yhat= ct.predict(Xtest)
df_test['Label'] = yhat
del df_test['TweetText']
df_test.head()
df_test.to_csv('submission.csv', index = False)


# In[ ]:




