#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[6]:


#Definieren der zwei Spaltenbenennungen
table_80 = pd.read_table('NYT80.csv', sep = ',', engine = 'python', nrows = 200000, names=['ngram', 'class'])
table_90 = pd.read_table('NYT90.csv', sep = ',', engine = 'python', nrows = 200000, names=['ngram', 'class'])
table_00 = pd.read_table('NYT00.csv', sep = ',', engine = 'python', nrows = 200000, names=['ngram', 'class'])


# In[7]:


table_80.to_csv('finalbigger_nyt.csv', index = False, mode = 'a')


# In[8]:


table_90.to_csv('finalbigger_nyt.csv', index = False, mode = 'a', header = None)


# In[9]:


table_00.to_csv('finalbigger_nyt.csv', index = False, mode = 'a', header = None)


# In[10]:


pd.read_csv('finalbigger_nyt.csv',nrows=20)


# In[1]:


import sys


# In[2]:


import csv


# In[3]:


maxInt = sys.maxsize


# In[4]:


while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


# In[ ]:




