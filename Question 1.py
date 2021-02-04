#!/usr/bin/env python
# coding: utf-8

# ### Battle of the Neighbourhoods - Toronto###

# Author: Nurdan Tekin
# 
# This notebook contains Questions 1, 2 & 3 of the Assignment. They have been segregated by Section headers

# In[1]:


import pandas as pd


# ## Question 1 ##

# ## Importing Data ##

# In[2]:


import requests


# In[3]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
wiki_url = requests.get(url)
wiki_url


# Response 200 means that we are able to make the connection to the page

# In[4]:


wiki_data = pd.read_html(wiki_url.text)
wiki_data


# In[5]:


len(wiki_data), type(wiki_data)


# We need the first table alone, so dropping the other tables

# In[6]:


wiki_data = wiki_data[0]
wiki_data


# Dropping Borough which are not assigned

# In[7]:


df = wiki_data[wiki_data["Borough"] != "Not assigned"]
df


# Grouping the records based on Postal Code

# In[8]:


df = df.groupby(['Postal Code']).head()
df


# Checking for number of records where Neighbourhood is "Not assigned"

# In[9]:


df.Neighbourhood.str.count("Not assigned").sum()


# In[10]:


df = df.reset_index()
df


# In[11]:


df.drop(['index'], axis = 'columns', inplace = True)
df


# In[12]:


df.shape


# Answer to Question 1: We have 103 rows and 3 columns

# In[ ]:




