#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#Einlesen der CSV-Datei
data = pd.read_csv("D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\Y90.csv")


# In[3]:


#5% des Dataframe
#rows = data.sample(frac =.05)


# In[3]:


#30% des Dataframe
rows = data.sample(frac =.3)


# In[5]:


#50% des Dataframe
rows = data.sample(frac =.5)


# In[5]:


#if (0.10*(len(data))== len(rows)):
    #print("Cool")
    #print(len(data), len(rows))


# In[ ]:


rows


# In[6]:


rows.to_csv(r"D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\50_percent\spiegel_quadgram_90_50_raw.csv", mode ='a', index = None)

