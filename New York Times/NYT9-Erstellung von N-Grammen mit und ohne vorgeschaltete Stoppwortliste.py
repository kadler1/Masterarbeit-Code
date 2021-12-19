#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.util import ngrams
import unicodedata
from collections import Counter
import csv
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize


# In[2]:


import sys
import csv
maxInt = sys.maxsize


# In[3]:


while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


# In[ ]:


nltk.download("stopwords")


# In[5]:


df = df.read_csv(r'D:/Studium/Master/Masterarbeit/Datenset_NYT/nyt_s/Grundlage/NYT_30_percent/NYT00_30.csv')


# In[9]:


#Definierung der Stoppwortliste
stop = stopwords.words('english')


# In[14]:


#Anwendung der Stoppwortliste - Auskommentieren bei Nicht-Nutzung
df['Article'] = chunk['Article'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))


# In[15]:


unigrams = (df['Article'].str.lower()
                .str.replace(r'[^a-z\s]', '')
                .str.split(expand=True)
                .stack())


# In[16]:


#Bigramm Erstellung
bigrams = unigrams + ' ' + unigrams.shift(-1)


# In[ ]:


#Trigramm Erstellung
trigrams = bigrams + ' ' + unigrams.shift(-2)


# In[ ]:


#Quadgramm Erstellung
quadgrams = bigrams + ' ' + bigrams.shift(-3)


# In[ ]:


#Quingramm Erstellung
quingram = bigrams + ' ' + trigrams.shift(-4)


# In[17]:


#Hinzufügen eines der N-Gramm Sets zu Dataframe
df['Article'] = bigrams.dropna().reset_index(drop=True)


# In[ ]:


display(df.head())


# In[19]:


#Auszählung der häufigsten N-Gramme
count = pd.Series(df['Article'].value_counts())[:20]


# In[ ]:


print(count)


# In[21]:


df.to_csv(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\Grundlage\NYT_30_percent\NGrams\NYT80_30_bigram.csv', index = False, mode = 'a', header = None)

