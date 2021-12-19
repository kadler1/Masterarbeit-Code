#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


#Einlesen der Datei
df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00.csv', sep=",", encoding='utf-8')


# In[ ]:


#Entfernen aller leerer Zeilen im Dataframe
df = df.dropna(how='all')


# In[ ]:


df.to_csv('outputfinal00_droped.csv', mode = 'a', index = None)

