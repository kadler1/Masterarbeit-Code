#!/usr/bin/env python
# coding: utf-8

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


# In[5]:


import csv


# In[6]:


import os


# In[7]:


inputFile = r"D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\Finale\finalcomplete_nyt.csv"


# In[8]:


outputFile = os.path.splitext(inputFile)[0] + "finalcomplete_modified_nyt.csv"


# In[9]:


#Änderung der Headerbenennungen
with open(inputFile, newline='', encoding ='utf-8') as inFile, open(outputFile, 'w', newline='', encoding ='utf-8') as outfile:
    r = csv.reader(inFile)
    w = csv.writer(outfile)
    next(r, None)
    w.writerow(['Article', 'Class'])
    for row in r:
        w.writerow(row)

