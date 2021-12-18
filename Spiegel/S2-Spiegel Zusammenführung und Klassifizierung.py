#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[ ]:


#Einlesen der CSV-Dateien, Definieren von 2 Spalten
table_80 = pd.read_table('D:\s80.csv', sep = ',', engine = 'python', chunksize = 10000, names=[1, 2])
table_90 = pd.read_table('D:\s90.csv', sep = ',', engine = 'python', chunksize = 10000, names=[1, 2])
table_00 = pd.read_table('D:\s00.csv', sep = ',', engine = 'python', chunksize = 10000, names=[1, 2])


# In[ ]:


#Dritte Spalte mit dem Wert "eighty" hinzufügen und speichern
for df in table_80:
    df[3] = 'eigthy'
    df.to_csv('Y80.csv', index = False, mode = 'a', header = None)


# In[ ]:


#Dritte Spalte mit dem Wert "ninety" hinzufügen und speichern
for df in table_90:
    df[3] =  'ninety'
    df.to_csv('Y90.csv', index = False, mode = 'a', header = None)


# In[ ]:


#Dritte Spalte mit dem Wert "zero" hinzufügen und speichern
for df in table_00:
    df[3] = 'zero'
    df.to_csv('Y00.csv', index = False, mode = 'a', header = None)


# In[6]:


#CSV-Dateien einelsen und Headernamen hinzufügen
table_80 = pd.read_table('Y80.csv', sep = ',', engine = 'python', chunksize = 20000, names=['ngram', 'number', 'class'])
table_90 = pd.read_table('Y90.csv', sep = ',', engine = 'python', chunksize = 20000, names=['ngram', 'number', 'class'])
table_00 = pd.read_table('Y00.csv', sep = ',', engine = 'python', chunksize = 20000, names=['ngram', 'number', 'class'])


# In[7]:


#Dataframe zur finalen CSV-Datei hinzufügen
for df in table_90:
    df.to_csv('final.csv', index = False, mode = 'a')


# In[8]:


#Dataframe zur finalen CSV-Datei hinzufügen
for df in table_80:
    df.to_csv('final.csv', index = False, mode = 'a', header = None)


# In[9]:


#Dataframe zur finalen CSV-Datei hinzufügen
for df in table_00:
    df.to_csv('final.csv', index = False, mode = 'a', header = None)


# In[3]:


pd.read_csv('final.csv',nrows=20)


# In[ ]:




