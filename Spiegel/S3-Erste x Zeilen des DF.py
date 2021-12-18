#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df1 = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel00_mini_nouns.csv')
df2 = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel80_mini_nouns.csv')
df3 = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel90_mini_nouns.csv')


# In[3]:


#Auswählen der oberen x Zeilen des Dataframe
df01 = df1.head(13087)
df02 = df2.head(13087)
df03 = df3.head(13087)
#df03 = df3.head(40478)
#df03 = df3.head(130874)


# In[4]:


#Dataframe als CSV-Datei
df01.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel00_mini_nouns2.csv', index = False, mode = 'a')
df02.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel80_mini_nouns2.csv', index = False, mode = 'a')
df03.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel90_mini_nouns2.csv', index = False, mode = 'a')


# In[ ]:




