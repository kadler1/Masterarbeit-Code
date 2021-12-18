#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


import nltk,csv,numpy


# In[3]:


import pandas as pd


# In[4]:


from nltk import word_tokenize, pos_tag, pos_tag_sents


# In[5]:


import itertools


# In[6]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# In[7]:


#Einlesen der CSV-Datei und Konvertierung der Spalte "Article" in Strings
data = pd.read_csv(r"D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\Grundlage\NYT_samples\NGrams\NYTfinal_5_quingram.csv", encoding='utf-8', converters = {'ngram': str})


# In[ ]:


data.head()


# In[9]:


#Definierung der Extraktion von Substantiven/ Substantiven und Adjektiven
def extract_noun(text):
    token = nltk.tokenize.word_tokenize(text, language='english')
    result = []
    for i in nltk.pos_tag(token):
        #if i[1].startswith('NN'):
        if i[1].startswith('NN'or 'JJ'):
            result.append(i[0])
    return(', '.join(result))


# In[10]:


#Anwendung der Funktion
data['Article'] = data['Article'].apply(extract_noun)


# In[ ]:


data['Article']


# In[12]:


#Funktion zur Umwandlung in Kleinbuchstaben 
lower = lambda x: x.lower()


# In[13]:


#Anwendung der Funktion
data['Article'] = data['Article'].apply(lower)


# In[ ]:


df1 = df.head(12968)


# In[ ]:


df1


# In[14]:


df1.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\spiegel00_5_nouns_adj2.csv', index = False, mode = 'a', header = None)


# In[15]:


table_00 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\spiegel00_5_nouns_adj2.csv', sep = ',', engine = 'python', names = ['ngram', 'class'])


# In[16]:


table_00['class'] = 'zero'


# In[18]:


table_00.to_csv('D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\spiegel00_5_nouns_adj.csv', index = False, mode = 'a', header = None)

