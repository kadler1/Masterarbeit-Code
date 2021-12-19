#!/usr/bin/env python
# coding: utf-8

# In[1]:


file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00.csv', 'r', encoding='utf-8')
#file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated1.csv', 'r', encoding='utf-8')
#file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated2.csv', 'r', encoding='utf-8')
#file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated3.csv', 'r', encoding='utf-8')
#file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated4.csv', 'r', encoding='utf-8')
#file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated5.csv', 'r', encoding='utf-8')
#file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated6.csv', 'r', encoding='utf-8')
#file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated7.csv', 'r', encoding='utf-8')


# In[2]:


file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated1.csv', 'w', encoding='utf-8')
#file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated2.csv', 'w', encoding='utf-8')
#file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated3.csv', 'w', encoding='utf-8')
#file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated4.csv', 'w', encoding='utf-8')
#file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated5.csv', 'w', encoding='utf-8')
#file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated6.csv', 'w', encoding='utf-8')
#file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated7.csv', 'w', encoding='utf-8')
#file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal00_updated8.csv', 'w', encoding='utf-8')


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('Photos of')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('Photo of')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('Photo essay of')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('EVENING HOURS')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('There is no Christmas')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('Photo')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('Photos')):
        print(line)
        file2.write(line)


# In[3]:


#Entfernung bestimmter Strings - einzelne Durchführung pro String erforderlich
for line in file1.readlines():
    if not  (line.startswith('Bill Cunningham Evening')):
        print(line)
        file2.write(line)


# In[4]:


file2.close()


# In[5]:


file1.close()

