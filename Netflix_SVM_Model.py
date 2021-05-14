import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import re

sw=stopwords.words('english')
lemma=WordNetLemmatizer()

pos_rev=pd.read_csv('C:/Users/admin/OneDrive/Desktop/MLDeployment/nlp_netflix_svm_model/data/pos.txt',sep='\n',header = None, encoding ='latin-1')

pos_rev['mood']=1.0

pos_rev=pos_rev.rename(columns={0:'review'})

neg_rev=pd.read_csv('C:/Users/admin/OneDrive/Desktop/MLDeployment/nlp_netflix_svm_model/data/neg.txt',sep='\n',header = None, encoding ='latin-1')

neg_rev['mood']=0.0

neg_rev=neg_rev.rename(columns={0:'review'})

#Cleaning up data
pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply(lambda x:x.lower())#convert to lower case
pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply(lambda x:re.sub(r'\d+[strdthnd]*\s','',x))#To remove digits and ordinal numbers
pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply(lambda x:re.sub(r'@\S+','',x))#To remove space and @
pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply\
(lambda x:x.translate(str.maketrans(dict.fromkeys(string.punctuation))))#To clean punctuation marks
pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply\
(lambda x:" ".join([lemma.lemmatize(word,pos='v') for word in x.split() if word not in (sw)]))#To clean stopwords and lemmatize

neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply(lambda x:x.lower())#convert to lower case
neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply(lambda x:re.sub(r'\d+[strdthnd]*\s','',x))#To remove digits and ordinal numbers
neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply(lambda x:re.sub(r'@\S+','',x))#To remove space and @
neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply\
(lambda x:x.translate(str.maketrans(dict.fromkeys(string.punctuation))))#To clean punctuation marks
neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply\
(lambda x:" ".join([lemma.lemmatize(word,pos='v') for word in x.split() if word not in (sw)]))#To clean stopwords

#Concatenating positive and negative review data
com_rev=pd.concat([pos_rev,neg_rev],axis=0).reset_index()#axis=0 as I want to combine rows of pos_rev and neg_rev.

#Train Test split
X_train,X_test,y_train,y_test=train_test_split(com_rev['review'].values,com_rev['mood'].values,test_size=0.2,random_state=101)

#Extra thing - create a dataframe instead of array
train_data=pd.DataFrame({'review':X_train,'mood':y_train})
test_data=pd.DataFrame({'review':X_test,'mood':y_test})

vectorizer=TfidfVectorizer()
train_vectors=vectorizer.fit_transform(train_data['review'])#making vectorizer learn the train data and create vocabulary and transform to numeric
test_vectors=vectorizer.transform(test_data['review'])#making vectorizer transform test vocabulary to numeric

from sklearn import svm
classifier=svm.SVC(kernel='linear')
classifier.fit(train_vectors,train_data['mood'])

#Predict Sentiment
#a=input('write the review: ')
#a=['best review']
# vector=vectorizer.transform(['worst review']).toarray()#.transform() needs input as a list
# my_pred=classifier.predict(vector)
# print(my_pred)

#Saving model
import joblib

model_file_name='NLP_netflix_svm_model.pkl'
joblib.dump(classifier,model_file_name)#Model object,filename

vectorizer_file_name='NLP_netfix_vectorizer.pkl'
joblib.dump(vectorizer,vectorizer_file_name)#Vectorizer object,filename