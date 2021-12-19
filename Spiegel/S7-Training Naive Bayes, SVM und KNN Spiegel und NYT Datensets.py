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
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


# In[7]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn import model_selection, svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[10]:


from nltk.tokenize import word_tokenize


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# In[14]:


import numpy as np


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[55]:


#Einlesen der CSV-Datei
#df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\30_percent\NGrams\spiegel_medium_quingram.csv')
df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\Grundlage\NYT_30_percent\NGrams\NYTfinal_30_quingram.csv')


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
plt.title("Class counts")
plt.show()


# In[59]:


le = LabelEncoder()


# In[60]:


df['Class_enc'] = le.fit_transform(df['Class'])


# In[ ]:


#Analyse des Datentyps
print(df.dtypes)


# In[63]:


df['word_count'] = df['Article'].str.split().str.len()


# In[ ]:


#Durchschnittliche Wortzählung pro Zeile pro Klasse
print(df.groupby('Class')['word_count'].mean())


# In[ ]:


sns.distplot(df[df['Class']=='eighty']['word_count'], label='eighty')
sns.distplot(df[df['Class']=='ninety']['word_count'], label='ninety'),
sns.distplot(df[df['Class']=='zero']['word_count'], label='zero'),
plt.legend()
plt.show()


# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords


# In[67]:


tokenized_messages = df['Article'].str.lower().apply(word_tokenize)


# In[68]:


X = df['Article']
y = df['Class_enc']


# In[69]:


#Trainings- und Testset Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)


# In[70]:


#Vectorizer
vectorizer = TfidfVectorizer(strip_accents='ascii')


# In[73]:


#Vektorisieren der Trainingsdaten
tfidf_train = vectorizer.fit_transform(X_train)


# In[76]:


#Vektorisieren der Testdaten
tfidf_test = vectorizer.transform(X_test)


# In[ ]:


#Parameterdefinierung für Naive Bayes Classifier
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)


# In[78]:


nb = MultinomialNB()


# In[ ]:


#Training des Naive Bayes Classifiers
nb.fit(tfidf_train, y_train)


# In[ ]:


#Ausgeben der Accuracy
print("Accuracy:",nb.score(tfidf_test, y_test))


# In[83]:


y_pred = nb.predict(tfidf_test)


# In[ ]:


#Ausgabe der missklassifizierten Daten
#misclassified = np.where(y_pred != y_test)
#print(misclassified)


# In[85]:


#file = open("sample.txt", "w")


# In[ ]:


#for input, prediction, label in zip (X_test, y_pred, y_test):
#        if prediction != label:
#            print(input, 'has been classified as ', prediction, 'and should be ', label)
#            file.write(input + "\n")


# In[87]:


#file.close()


# In[ ]:


#Erstellung der Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n")
print(cm)


# In[ ]:


#Erstellung des Klassifikationsreports
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)


# In[50]:


y_pred_proba = nb.predict(tfidf_test)


# In[51]:


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


#Training des SVM Modells
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(tfidf_train,y_train)
# Vorhersage der Klassenlabels mit dem Validierungsset
predictions_SVM = SVM.predict(tfidf_test)
# Ermittlung der Accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)


# In[45]:


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


# In[57]:


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


knn = KNeighborsClassifier(n_neighbors=7)
#Anwendung des kNN Classifiers
clf = knn.fit(tfidf_train, y_train)
predicted = knn.predict(tfidf_test)
# Ermittlung der Accuracy
print("Knn Accuracy Score -> ",accuracy_score(predicted, y_test)*100)


# In[60]:


y_pred = knn.predict(tfidf_test)


# In[ ]:


#Erstellung des Klassifikationsreports
cr = classification_report(y_test, y_pred)
print("\n\nClassification Report\n")
print(cr)


# In[62]:


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

