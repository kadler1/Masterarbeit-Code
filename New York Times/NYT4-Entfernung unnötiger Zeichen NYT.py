#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import string


# In[3]:


import sys


# In[4]:


import csv


# In[5]:


maxInt = sys.maxsize


# In[6]:


while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
            maxInt = int(maxInt/10)


# In[7]:


master_table = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\Grundlage\NYT00.csv', sep =',', engine = 'python', encoding = 'utf-8', names=['Article', 'Class'])


# In[8]:


master_table['Article'] = master_table['Article'].str.replace('\"','')


# In[9]:


master_table['Article'] = master_table['Article'].str.replace('\'','')


# In[16]:


master_table['Article'] = master_table['Article'].str.replace('\:','')


# In[17]:


master_table['Article'] = master_table['Article'].str.replace('\-','')


# In[18]:


master_table['Article'] = master_table['Article'].str.replace('\.','')


# In[19]:


master_table['Article'] = master_table['Article'].str.replace('\]','')


# In[ ]:


master_table['Article'] = master_table['Article'].str.replace('\[','')


# In[10]:


result = master_table.head(10)


# In[ ]:


print(result)


# In[12]:


master_table.to_csv(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\Grundlage\NYT00_new.csv', mode = 'a', index = None)


# In[ ]:




