#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


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


nltk.download('wordnet')


# In[10]:


from sklearn.naive_bayes import MultinomialNB


# In[11]:


from sklearn import model_selection, svm, datasets


# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# In[17]:


import numpy as np


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[20]:


#Einlesen der CSV-Datei
#Read CSV Sample Set NYT
df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\Grundlage\NYT_50_percent\NYTfinal_50_nouns.csv')


# In[21]:


df.head()


# In[22]:


display(df.describe())


# In[23]:


#Zählung Dataframe Zeilen
print(df['Class'].value_counts())


# In[24]:


#Prozentuelle Datenverteilung nach Klassen
print(df['Class'].value_counts(normalize=True))


# In[25]:


#Visuelle Darstellung der Datenverteilung nach Klassen
sns.countplot(df['Class'])
plt.title("Class Counts")
plt.show()


# In[26]:


le = LabelEncoder()


# In[27]:


#Anwendung des Labelencoders
df['Class_enc'] = le.fit_transform(df['Class'])


# In[29]:


#Analyse des Datentyps
print(df.dtypes)


# In[30]:


df['word_count'] = df['Article'].str.split().str.len()


# In[31]:


#Durchschnittliche Wortzählung pro Zeile pro Klasse
print(df.groupby('Class')['word_count'].mean())


# In[33]:


#Funktion des Porter Stemmer
def porter(tokens):
    stemmer = PorterStemmer()
    stemmed = []
    for token in tokens:
            stemmed.append(stemmer.stem(token))
    return " ".join(stemmed)


# In[34]:


#Tokenisierung
tokenized_messages = df['Article'].str.lower().apply(word_tokenize)


# In[35]:


#Anwendung des Porter Stemmers
tokenized_messages = tokenized_messages.apply(porter)


# In[36]:


df['Article'] = tokenized_messages


# In[37]:


display(df.head())


# In[48]:


X = df['Article']
y = df['Class_enc']


# In[49]:


#Trainings- und Testset Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)


# In[50]:


#Vectorizer
vectorizer = TfidfVectorizer(strip_accents='ascii')


# In[51]:


#Vektorisieren der Trainingsdaten
tfidf_train = vectorizer.fit_transform(X_train)


# In[53]:


#Vektorisieren der Testdaten
tfidf_test = vectorizer.transform(X_test)


# In[54]:


nb = MultinomialNB()


# In[55]:


#Training des Naive Bayes Classifiers
nb.fit(tfidf_train, y_train)


# In[56]:


y_pred = nb.predict(tfidf_test)


# In[57]:


#Ausgabe der Accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(y_pred, y_test)*100)


# In[58]:


#Erstellung der Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n")
print(cm)


# In[59]:


#Erstellung des Klassifikationsreports
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)


# In[60]:


category_id_df = df[['Class', 'Class_enc']].drop_duplicates().sort_values('Class_enc')


# In[61]:


#Visualisierung der Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Class.values, yticklabels=category_id_df.Class.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[62]:


#Check Over- & Underfitting
print('Training set score: {:.4f}'.format(nb.score(tfidf_train, y_train)))
print('Test set score: {:.4f}'.format(nb.score(tfidf_test, y_test)))


# In[ ]:


#Training des SVM Modells
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(tfidf_train,y_train)
# Vorhersage der Klassenlabels mit dem Validierungsset
predictions_SVM = SVM.predict(tfidf_test)
# Ermittlung der Accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)


# In[ ]:


y_pred = SVM.predict(tfidf_test)


# In[ ]:


#Erstellung des Klassifikationsreports
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)


# In[ ]:


#Erstellung der Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n")
print(cm)


# In[ ]:


category_id_df = df[['Class', 'Class_enc']].drop_duplicates().sort_values('Class_enc')


# In[ ]:


#Visualisierung der Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Class.values, yticklabels=category_id_df.Class.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[ ]:


#Check Over- & Underfitting
print('Training set score: {:.4f}'.format(SVM.score(tfidf_train, y_train)))
print('Test set score: {:.4f}'.format(SVM.score(tfidf_test, y_test)))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=7)
#Anwendung des kNN Classifiers
clf = knn.fit(tfidf_train, y_train)
predicted = knn.predict(tfidf_test)
#Ermittlung der Accuracy
print("Knn Accuracy Score -> ",accuracy_score(predicted, y_test)*100)


# In[ ]:


y_pred = knn.predict(tfidf_test)


# In[ ]:


#Erstellung des Klassifikationsreports
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)


# In[ ]:


category_id_df = df[['Class', 'Class_enc']].drop_duplicates().sort_values('Class_enc')


# In[ ]:


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

