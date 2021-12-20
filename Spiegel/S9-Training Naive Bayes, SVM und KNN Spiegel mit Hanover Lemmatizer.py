#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
from collections import Counter


# In[3]:


import seaborn as sns


# In[4]:


import nltk


# In[5]:


from nltk.stem import WordNetLemmatizer


# In[6]:


from nltk.stem.snowball import SnowballStemmer


# In[7]:


from nltk.stem.porter import PorterStemmer


# In[8]:


from nltk.tokenize import word_tokenize


# In[9]:


from HanTa import HanoverTagger as ht


# In[10]:


#Deutscher Lemmatizer
tagger = ht.HanoverTagger('morphmodel_ger.pgz')


# In[11]:


#nltk.download('wordnet')


# In[12]:


from sklearn.naive_bayes import MultinomialNB


# In[13]:


from sklearn import model_selection, svm, datasets


# In[14]:


from sklearn.neighbors import KNeighborsClassifier


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# In[19]:


import numpy as np


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[54]:


#Einlesen der CSV-Datei
df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel_mini_quingram.csv')


# In[ ]:


df.head()


# In[ ]:


display(df.describe())


# In[ ]:


#Zählung Dataframe Zeilen
print(df['Class'].value_counts())


# In[ ]:


#Prozentuelle Datenverteilung nach Klassen
print(df['Class'].value_counts(normalize=True))


# In[ ]:


#Visuelle Darstellung der Datenverteilung nach Klassen
sns.countplot(df['Class'])
plt.title("Class Counts")
plt.show()


# In[60]:


le = LabelEncoder()


# In[61]:


#Anwendung des Labelencoders
df['Class_enc'] = le.fit_transform(df['Class'])


# In[ ]:


#Analyse des Datentyps
print(df.dtypes)


# In[64]:


df['word_count'] = df['Article'].str.split().str.len()


# In[ ]:


#Durchschnittliche Wortzählung pro Zeile pro Klasse
print(df.groupby('Class')['word_count'].mean())


# In[77]:


#Funktion des Lemmatizers
def lemmatize(tokens):
    #lemmatizer = tagger.analyze()
    lemmatized = []
    finish = []
    for token in tokens:
            lemmatized.append(tagger.analyze(token))
    for inner in lemmatized:
        finish.append(inner[0])
    return " ".join(finish)


# In[78]:


#Tokenisierung
tokenized_messages = df['Article'].str.lower().apply(word_tokenize)


# In[79]:


#Anwendung des Lemmatizers
tokenized_messages = tokenized_messages.apply(lemmatize)


# In[82]:


#Umwandlung in Kleinbuchstaben
df['Article'] = tokenized_messages.str.lower()


# In[ ]:


display(df.head(50))


# In[84]:


X = df['Article']
y = df['Class_enc']


# In[85]:


#Trainings- und Testset Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)


# In[86]:


#Vectorizer
vectorizer = TfidfVectorizer(strip_accents='ascii')


# In[87]:


#Vektorisieren der Trainingsdaten
tfidf_train = vectorizer.fit_transform(X_train)


# In[89]:


#Vektorisieren der Testdaten
tfidf_test = vectorizer.transform(X_test)


# In[90]:


nb = MultinomialNB()


# In[91]:


#Training des Naive Bayes Classifiers
nb.fit(tfidf_train, y_train)


# In[ ]:


#Ausgeben der Accuracy
print("Accuracy:",nb.score(tfidf_test, y_test))


# In[93]:


y_pred = nb.predict(tfidf_test)


# In[94]:


#Erstellung der Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n")
print(cm)


# In[95]:


#Erstellung des Klassifikationsreports
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)


# In[62]:


category_id_df = df[['Class', 'Class_enc']].drop_duplicates().sort_values('Class_enc')


# In[63]:


#Visualisierung der Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Class.values, yticklabels=category_id_df.Class.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[64]:


#Check Over- & Underfitting
print('Training set score: {:.4f}'.format(nb.score(tfidf_train, y_train)))
print('Test set score: {:.4f}'.format(nb.score(tfidf_test, y_test)))


# In[65]:


#Training  des Support Vector Machine Modells
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(tfidf_train,y_train)
# Vorhersage der Klassenlabels mit dem Validierungsset
predictions_SVM = SVM.predict(tfidf_test)
# Ermittlung der Accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)


# In[66]:


y_pred = SVM.predict(tfidf_test)


# In[67]:


#Erstellung des Klassifikationsreports
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)


# In[68]:


#Erstellung der Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n")
print(cm)


# In[69]:


category_id_df = df[['Class', 'Class_enc']].drop_duplicates().sort_values('Class_enc')


# In[70]:


#Visualisierung der Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Class.values, yticklabels=category_id_df.Class.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[71]:


knn = KNeighborsClassifier(n_neighbors=7)
#Anwendung des kNN Classifiers
clf = knn.fit(tfidf_train, y_train)
predicted = knn.predict(tfidf_test)
# Ermittlung der Accuracy
print("Knn Accuracy Score -> ",accuracy_score(predicted, y_test)*100)


# In[72]:


y_pred = knn.predict(tfidf_test)


# In[73]:


#Erstellung des Klassifikationsreports
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)


# In[74]:


category_id_df = df[['Class', 'Class_enc']].drop_duplicates().sort_values('Class_enc')


# In[75]:


#Visualisierung der Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Class.values, yticklabels=category_id_df.Class.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[ ]:


#Save the model
#from sklearn.externals import joblib
#joblib.dump(clf, 'filename.pkl')
#model = joblib.load('filename.pkl')

