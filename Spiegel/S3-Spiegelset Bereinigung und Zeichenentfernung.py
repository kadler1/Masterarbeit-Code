#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import regex as re
import sys
import csv
maxInt = sys.maxsize


# In[2]:


while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


# In[3]:


#master_table = pd.read_csv('D:\Studium\Master\Masterarbeit\Datensets_Spiegel\spiegel_s\spiegel_sample\spiegel80_5_sample.csv', sep =',', engine = 'python', encoding = 'utf-8', names=['ngram', 'number', 'class'])
master_table = pd.read_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\50_percent\spiegel_quingram_00_huge_raw2.csv', sep =',', engine = 'python', skiprows = [i for i in range(1,69)], encoding = 'utf-8')


# In[5]:


master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã‚Â¶','ö')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã‚Â¼','ü')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã‚Â¤','ä')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã…â€œ','Ü')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã…Â¸','ß')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾','Ä')


# In[6]:


master_table['Article'] = master_table['Article'].str.replace('\ÃƒÂ¤','ä')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÂ¼','ü')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÅ','ß')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÂ¶','ö')


# In[7]:


master_table['Article'] = master_table['Article'].str.replace('\"','')


# In[8]:


master_table['Article'] = master_table['Article'].str.replace('\'','')


# In[9]:


master_table['Article'] = master_table['Article'].str.replace('\;','')


# In[10]:


master_table['Article'] = master_table['Article'].str.replace('\,','')


# In[11]:


master_table['Article'] = master_table['Article'].str.replace('\:','')


# In[12]:


master_table['Article'] = master_table['Article'].str.replace('\(','')


# In[13]:


master_table['Article'] = master_table['Article'].str.replace('\)','')


# In[14]:


master_table['Article'] = master_table['Article'].str.replace('\-',' ')


# In[15]:


master_table['Article'] = master_table['Article'].str.replace('\.','')


# In[16]:


master_table['Article'] = master_table['Article'].str.replace('\?','')


# In[17]:


master_table['Article'] = master_table['Article'].str.replace('\!','')


# In[18]:


master_table['Article'] = master_table['Article'].str.replace('\*','')


# In[19]:


master_table['Article'] = master_table['Article'].str.replace('\/','')


# In[20]:


master_table['Article'] = master_table['Article'].str.replace(' \ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â±','ñ')


# In[21]:


master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©','e')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â','i')


# In[22]:


master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾','Ä')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã¢â‚¬Å¾','Ä')
master_table['Article'] = master_table['Article'].str.replace('\ssâ€œ','Ü')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©','e')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â','i')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³','o')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™ÃƒÂ¯Ã‚ÂÃƒÆ’Ã†â€™ÃƒÂ¯Ã‚Â','')


# In[23]:


master_table['Article'] = master_table['Article'].str.replace('\[','')
master_table['Article'] = master_table['Article'].str.replace('\*','')
master_table['Article'] = master_table['Article'].str.replace('\]','')
master_table['Article'] = master_table['Article'].str.replace('\Â¸','')
master_table['Article'] = master_table['Article'].str.replace('\/','')
master_table['Article'] = master_table['Article'].str.replace('\\','')
master_table['Article'] = master_table['Article'].str.replace('\!','')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã†â€™Ãƒâ€¹Ã…â€œ','Ö')
master_table['Article'] = master_table['Article'].str.replace('\¿Ã‚Â½','A')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§','')
master_table['Article'] = master_table['Article'].str.replace('\%','')


# In[24]:


master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã‚Â©','e')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã‚Â§','c')
master_table['Article'] = master_table['Article'].str.replace('\|','')
master_table['Article'] = master_table['Article'].str.replace('\&','')
master_table['Article'] = master_table['Article'].str.replace('\ÃƒÆ’Ã‚Â­','i')
master_table['Article'] = master_table['Article'].str.replace('NUM','')
master_table['Article'] = master_table['Article'].str.replace('  ',' ')


# In[ ]:


master_table.to_csv(r'D:\Studium\Master\Masterarbeit\Datensets_Spiegel\Spiegel_s\50_percent\spiegel_quingram_00_huge_clean.csv', mode = 'a', index = None)

