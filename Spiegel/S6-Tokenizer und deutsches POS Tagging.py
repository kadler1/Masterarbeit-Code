#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk import word_tokenize


# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


from HanTa import HanoverTagger as ht


# In[5]:


#Definieren des Hanover Taggers
tagger = ht.HanoverTagger('morphmodel_ger.pgz')


# In[6]:


#Einlesen der CSV-Datei und Konvertierung der Spalte "Article" zu String
data = pd.read_csv(r"D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\Grundlage\NYT_samples\NGrams\finalNYT5_sample_quadgram.csv", encoding='utf-8', converters = {'Article': str})


# In[7]:


data.head()


# In[8]:


#Definieren der zu extrahierenden - alle Substantive/ alle Substantive und Adjektive
def extract_noun(tokens):
    tagged = []
    result = []
    for token in tokens:
        tagged.append(tagger.analyze(token))
    for inner in tagged:
        if inner[1].startswith('N'):
        #if inner[1].startswith('N'or'ADJ'):
            result.append(inner[0])
        #if inner[1].startswith('ADJ'):
            #result.append(inner[0])
    return(', '.join(result))


# In[9]:


#Tokenisierung
data['Article'] = data['Article'].apply(word_tokenize)


# In[10]:


#Anwendung der POS-Tagging Funktion
data['Article'] = data['Article'].apply(extract_noun)


# In[13]:


df = data["Article"].str.cat(sep=' ').split()


# In[7]:


#Funktion zur Umwandlung in Kleinbuchstaben
lower = lambda x: x.lower()


# In[ ]:


#Anwenden der Funktion
data['Article'] = data['Article'].apply(lower)


# In[ ]:


#Bereinigen einer Unreinheit
data['Article'] = data['Article'].str.replace('\ãƒâ€ž','ä')


# In[ ]:


#data = data.head(12968)
#data = data.head(39263)


# In[9]:


#Ausschreiben von leeren Zeilen
#df = data['Article'].replace('', np.nan, inplace=True)
data['Article'] = data['Article'].replace('', np.nan, inplace=True)


# In[6]:


#Nur Zeilen mit Inhalt behalten
#df = data['Article'].dropna()
data['Article'] = data['Article'].dropna()


# In[ ]:


#Definition der Dataframe Größe
#df1 = df.head(12968)
#df1 = df.head(39263)
#df1 = df.head(301995)
#df1 = df.head(183027)
df1 = df.head(40478)


# In[ ]:


df1.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\30_percent\spiegel00_medium_nouns_adj2.csv', index = False, mode = 'a')


# In[ ]:


table_00 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\30_percent\spiegel00_medium_nouns_adj2.csv', sep = ',', engine = 'python', encoding = 'utf-8', names = ['Article', 'Class'])


# In[ ]:


table_00['Class'] = 'zero'


# In[ ]:


table_00


# In[ ]:


table_00.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\30_percent\spiegel00_medium_nouns_adj.csv', index = False, mode = 'a')

