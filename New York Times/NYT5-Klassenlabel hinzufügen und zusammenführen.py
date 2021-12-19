#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


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


# In[6]:


#Öffnen der Jahrzehnt-Datensets 
table_80 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_ready.csv', sep = ',', engine = 'python', chunksize = 10000, names=[1])
table_90 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal90_ready.csv', sep = ',', engine = 'python', chunksize = 10000, names=[1])
table_00 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinalnew80_ready.csv', sep = ',', engine = 'python', chunksize = 10000, names=[1])


# In[7]:


#Einfügen einer zweiten Zeile mit dem Wert "eighty"
for df in table_80:
    df[2] = 'eigthy'
    df.to_csv('NYT80.csv', index = False, mode = 'a', header = None)


# In[8]:


#Einfügen einer zweiten Zeile mit dem Wert "ninety"
for df in table_90:
    df[2] =  'ninety'
    df.to_csv('NYT90.csv', index = False, mode = 'a', header = None)


# In[9]:


#Einfügen einer zweiten Zeile mit dem Wert "zero"
for df in table_00:
    df[2] = 'zero'
    df.to_csv('NYT00.csv', index = False, mode = 'a', header = None)


# In[10]:


#table_80 = pd.read_table('NYT80.csv', sep = ',', engine = 'python', chunksize = 20000, names=['ngram', 'class'])
#table_90 = pd.read_table('NYT90.csv', sep = ',', engine = 'python', chunksize = 20000, names=['ngram', 'class'])
#table_00 = pd.read_table('NYT00.csv', sep = ',', engine = 'python', chunksize = 20000, names=['ngram', 'class'])


# In[11]:


#for df in table_80:
#    df.to_csv('final.csv', index = False, mode = 'a')


# In[12]:


#for df in table_90:
#    df.to_csv('final.csv', index = False, mode = 'a', header = None)


# In[13]:


#for df in table_00:
#    df.to_csv('final.csv', index = False, mode = 'a', header = None)


# In[14]:


#pd.read_csv('final.csv',nrows=20)

