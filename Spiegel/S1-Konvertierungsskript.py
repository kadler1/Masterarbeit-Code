#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


tsv_cont = 'D:\s00.tsv'


# In[3]:


csv_table = pd.read_table(tsv_cont, sep=r'\t', engine='python', chunksize = 10000)


# In[4]:


#Konvertierung der TSV-Dateien zu CSV-Dateien
for df in csv_table:
    df.to_csv('s00.csv', index = False, mode = 'a')


# In[ ]:




