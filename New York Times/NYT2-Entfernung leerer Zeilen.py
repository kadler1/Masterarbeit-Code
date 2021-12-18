#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#Einlesen der Datei
df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal90 - Kopie.csv', encoding='utf-8')


# In[3]:


#Entfernen aller leerer Zeilen im Dataframe
df = df.dropna(how='all')


# In[4]:


df.to_csv('outputfinal90_droped.csv', mode = 'a', index = None)

