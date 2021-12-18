#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from nltk.corpus import stopwords


# In[2]:


df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\50_percent\spiegel_quingram_00_huge_clean.csv', sep =',', engine = 'python', encoding = 'utf-8')


# In[4]:


#Definition der Stopwortlisten-Anwendung / auskommentieren bei Nicht-Nutzung der Stoppwortliste
stop = stopwords.words('german')


# In[5]:


#Anwendung Stopwortliste / auskommentieren bei Nicht-Nutzung der Stoppwortliste
df['Article'] = df['Article'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))


# In[6]:


#print(df)


# In[7]:


#Nur Zeilen mit 2/3/4/5-Grammen behalten
mask = df['Article'].str.strip().str.split(' ').str.len().eq(5)


# In[8]:


out = df[mask]


# In[ ]:


print(out.head)


# In[10]:


#out.to_csv('D:\Studium\Master\Masterarbeit\Datensets_Spiegel\spiegel_s\spiegel_sample\spiegel_final_5_quadgram.csv', mode = 'a', index = None)


# In[11]:


#df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\spiegel_s\spiegel_sample\tworowspiegel_sample_quadgram.csv', sep =',', engine = 'python')


# In[12]:


#Nur die ersten Zeilen behalten für gleiche Datengrößen
df1 = out.head(301995)
#df1 = out.head(183027)
#df1 = out.head(39263)


# In[ ]:


#Zeile Anzahl entfernen
df1.drop(df.columns[[1]], axis=1, inplace=True)


# In[ ]:


#Nur Kleinschreibung
df1['Article'] = df1['Article'].str.lower()


# In[18]:


df1.to_csv(r"D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\50_percent\spiegel_quingram_00_big.csv", mode ='a', index = None)


# In[ ]:




