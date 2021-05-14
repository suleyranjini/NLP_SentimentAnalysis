#Loading model
import joblib

vect=joblib.load('NLP_netfix_vectorizer.pkl')
clf=joblib.load('NLP_netflix_svm_model.pkl')

#prediction

vector=vect.transform(['excellent review'])
#my_pred=classifier.predict(vector)
print(clf.predict(vector))
