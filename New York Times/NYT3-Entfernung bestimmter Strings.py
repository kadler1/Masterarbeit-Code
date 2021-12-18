#!/usr/bin/env python
# coding: utf-8

# In[1]:


file1 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal90_updated6.csv', 'r', encoding='utf-8')


# In[2]:


file2 = open(r'D:\Studium\Master\Masterarbeit\Datenset_NYT\nyt_s\outputfinal90_updated7.csv', 'w', encoding='utf-8')


# In[ ]:


#Entfernung inhaltsloser Artikel
for line in file1.readlines():
    if not  (line.startswith('Photo')):
        print(line)
        file2.write(line)


# In[ ]:


for line in file1.readlines():
    if not  (line.startswith('Photos')):
        print(line)
        file2.write(line)


# In[ ]:


for line in file1.readlines():
    if not  (line.startswith('THIS WEEK')):
        print(line)
        file2.write(line)


# In[ ]:


for line in file1.readlines():
    if not  (line.startswith('SOAPBOX')):
        print(line)
        file2.write(line)


# In[ ]:


for line in file1.readlines():
    if not  (line.startswith('NEIGHBORHOOD REPORT')):
        print(line)
        file2.write(line)


# In[4]:


file2.close()


# In[5]:


file1.close()


# In[ ]:




