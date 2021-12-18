#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


import csv


# In[3]:


import itertools


# In[4]:


from xml.etree import ElementTree as ET


# In[5]:


#get the path where the data lays for every decade separate
path = "D:/Studium/Master/Masterarbeit/Datenset_NYT/nyt_corpus_structured/00/"


# In[6]:


#prepare output file
file = open('outputfinal00.csv', 'w', encoding='utf-8')


# In[ ]:


#go through all directories and subdirectories, find .xml files, extract the articles in element
for dirpath, dirs, files in os.walk(path):
    for filename in files:
        if not filename.endswith(".xml"): continue
        if filename.endswith(".xml"):
            fullname = os.path.join(dirpath, filename)
            tree = ET.parse(fullname)
            print (tree)
            root = tree.getroot()
            for row in root.iter('p'):
                try:
                    file.write(row.text + ",")
                except:
                    pass
            file.write("\n")
            #if row == None:
                #continue
            #file.write("\n")


# In[8]:


file.close()

