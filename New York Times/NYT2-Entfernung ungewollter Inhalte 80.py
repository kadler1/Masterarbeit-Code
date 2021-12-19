#!/usr/bin/env python
# coding: utf-8

# In[1]:


file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal80.csv', 'r', encoding='utf-8')
file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal80_updated1.csv', 'r', encoding='utf-8')
file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal80_updated2.csv', 'r', encoding='utf-8')


# In[2]:


file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal80_updated1.csv', 'w', encoding='utf-8')
file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal80_updated2.csv', 'w', encoding='utf-8')
file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal80_updated3.csv', 'w', encoding='utf-8')


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('LEAD:*3*** COMPANY REPORTS')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('LEAD: To the Sports Editor:')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('CURRENTS')):
        print(line)
        file2.write(line)


# In[4]:


file2.close()


# In[5]:


file1.close()

