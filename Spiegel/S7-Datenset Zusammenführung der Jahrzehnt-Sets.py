#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import sys


# In[3]:


import csv


# In[4]:


maxInt = sys.maxsize


# In[5]:


while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


# In[6]:


#table_00 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\spiegel00_5.csv', sep = ',', engine = 'python', names=['ngram', 'number', 'class'])
#table_80 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\spiegel80_5.csv', sep = ',', engine = 'python', names=['ngram', 'number', 'class'])
#table_90 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\spiegel90_5.csv', sep = ',', engine = 'python', names=['ngram', 'number', 'class'])
table_00 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel00_mini_nouns_adj2.csv', sep = ',', engine = 'python', encoding = 'utf-8')
table_80 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel80_mini_nouns_adj2.csv', sep = ',', engine = 'python', encoding = 'utf-8')
table_90 = pd.read_table(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel90_mini_nouns_adj2.csv', sep = ',', engine = 'python', encoding = 'utf-8')


# In[8]:


table_80.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel_mini_nouns_adj.csv', index = False, mode = 'a')
table_90.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel_mini_nouns_adj.csv', index = False, mode = 'a', header = None)
table_00.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel_mini_nouns_adj.csv', index = False, mode = 'a', header = None)


# In[9]:


#table_90.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\30_percent\NGrams\spiegel_medium_bigram.csv', index = False, mode = 'a', header = None)


# In[10]:


#table_00.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\30_percent\NGrams\spiegel_medium_bigram.csv', index = False, mode = 'a', header = None)


# In[11]:


df = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\spiegel_sample\NGrams\spiegel_mini_nouns_adj.csv', sep = ',', engine = 'python', encoding = 'utf-8')

