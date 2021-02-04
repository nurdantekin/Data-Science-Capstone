#!/usr/bin/env python
# coding: utf-8

# ### Battle of the Neighbourhoods - Toronto###

# Author: Nurdan Tekin
# 
# This notebook contains Questions 1, 2 & 3 of the Assignment. They have been segregated by Section headers

# In[51]:


import pandas as pd


# ## Question 1 ##

# ## Importing Data ##

# In[52]:


import requests


# In[53]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
wiki_url = requests.get(url)
wiki_url


# Response 200 means that we are able to make the connection to the page

# In[54]:


wiki_data = pd.read_html(wiki_url.text)
wiki_data


# In[55]:


len(wiki_data), type(wiki_data)


# We need the first table alone, so dropping the other tables

# In[56]:


wiki_data = wiki_data[0]
wiki_data


# Dropping Borough which are not assigned

# In[57]:


df = wiki_data[wiki_data["Borough"] != "Not assigned"]
df


# Grouping the records based on Postal Code

# In[58]:


df = df.groupby(['Postal Code']).head()
df


# Checking for number of records where Neighbourhood is "Not assigned"

# In[59]:


df.Neighbourhood.str.count("Not assigned").sum()


# In[60]:


df = df.reset_index()
df


# In[61]:


df.drop(['index'], axis = 'columns', inplace = True)
df


# In[62]:


df.shape


# Answer to Question 1: We have 103 rows and 3 columns

# ### Question 2 ###

# Installing geocoder

# In[63]:


pip install geocoder


# In[64]:


import geocoder # import geocoder


# geocoder wasn't working that's why using csv file given in the course

# In[65]:


data = pd.read_csv("https://cocl.us/Geospatial_data")
data


# In[66]:


print("The shape of our wiki data is: ", df.shape)
print("the shape of our csv data is: ", data.shape)


# Since the dimensions are the same, we can try to join on the postal codes to get the required data.
# 
# Checking the column types of both the dataframes, especially Postal Code column since we are trying to join on it

# In[67]:


df.dtypes


# In[68]:


data.dtypes


# In[69]:


combined_data = df.join(data.set_index('Postal Code'), on='Postal Code', how='inner')
combined_data


# In[70]:


combined_data.shape


# Solution: We get 103 rows as expected when we do a inner join, so we have good data.
