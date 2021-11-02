#!/usr/bin/env python
# coding: utf-8

# # spam detection

# In[3]:


from sklearn.naive_bayes import BernoulliNB ,GaussianNB ,MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


# In[5]:


spam = pd.read_csv('spam.csv')
spam.head()

#creating spam column as target 
spam["Spam"] = spam.Category.apply(lambda x: 1 if x== 'spam' else 0)
spam.head()


# In[6]:


#vectorize the messsage
X = spam.Message
y = spam.Spam


# In[7]:


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = .1)


# In[8]:


vector = CountVectorizer()
Xtrain_vector = vector.fit_transform(Xtrain.values).toarray()
# t=X.toarray()
# t[2]
Xtrain_vector[:3]


# In[9]:


X_vector = vector.transform(X).toarray()


# In[10]:


#model building and training
model1 = BernoulliNB()
model2 = MultinomialNB()
model3 = GaussianNB()

model1.fit(Xtrain_vector,ytrain)
model2.fit(Xtrain_vector,ytrain)
model3.fit(Xtrain_vector,ytrain)

# tts1 = model1.score(vector.transform(Xtest).toarray(),ytest)
# tts2 = model2.score(vector.transform(Xtest).toarray(),ytest)
# tts3 = model3.score(vector.transform(Xtest).toarray(),ytest)

# score1 = cross_val_score(model1,X_vector,y,cv = 2)
# score2 = cross_val_score(model2,X_vector,y,cv = 2)
# score3 = cross_val_score(model3,X_vector,y,cv = 2)


# data = { "Bernoulli":[tts1,score1.mean()]
#        ,'Multinomial':[tts2,score2.mean()],
#             'Gaussian':[tts3,score3.mean()]}
# da = pd.DataFrame(data,index=['Train test split','Cross Validation'])

# da


# # using above thing with PIPELINE

# In[11]:


from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('vectorizer',CountVectorizer()),
    ('model1',model1)
])


# In[12]:


pipe.fit(Xtrain,ytrain)


# In[ ]:





# In[13]:


pipe.score(Xtest,ytest)


# In[14]:


pickle.dump(pipe,open('modeldone.pkl','wb'))


# In[16]:


m=pickle.load(open('modeldone.pkl','rb'))


# In[18]:


m.score(Xtest,ytest)


# In[ ]:




