#!/usr/bin/env python
# coding: utf-8

# ### Battle of the Neighbourhoods - Toronto###

# Author: Nurdan Tekin
# 
# This notebook contains Questions 1, 2 & 3 of the Assignment. They have been segregated by Section headers

# In[3]:


import pandas as pd


# ## Question 1 ##

# ## Importing Data ##

# In[4]:


import requests


# In[5]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
wiki_url = requests.get(url)
wiki_url


# Response 200 means that we are able to make the connection to the page

# In[6]:


wiki_data = pd.read_html(wiki_url.text)
wiki_data


# In[7]:


len(wiki_data), type(wiki_data)


# We need the first table alone, so dropping the other tables

# In[8]:


wiki_data = wiki_data[0]
wiki_data


# Dropping Borough which are not assigned

# In[9]:


df = wiki_data[wiki_data["Borough"] != "Not assigned"]
df


# Grouping the records based on Postal Code

# In[10]:


df = df.groupby(['Postal Code']).head()
df


# Checking for number of records where Neighbourhood is "Not assigned"

# In[11]:


df.Neighbourhood.str.count("Not assigned").sum()


# In[12]:


df = df.reset_index()
df


# In[13]:


df.drop(['index'], axis = 'columns', inplace = True)
df


# In[14]:


df.shape


# Answer to Question 1: We have 103 rows and 3 columns

# ### Question 2 ###

# Installing geocoder

# In[13]:


pip install geocoder


# In[15]:


import geocoder # import geocoder


# geocoder wasn't working that's why using csv file given in the course

# In[16]:


data = pd.read_csv("https://cocl.us/Geospatial_data")
data


# In[17]:


print("The shape of our wiki data is: ", df.shape)
print("the shape of our csv data is: ", data.shape)


# Since the dimensions are the same, we can try to join on the postal codes to get the required data.
# 
# Checking the column types of both the dataframes, especially Postal Code column since we are trying to join on it

# In[18]:


df.dtypes


# In[19]:


data.dtypes


# In[20]:


combined_data = df.join(data.set_index('Postal Code'), on='Postal Code', how='inner')
combined_data


# In[21]:


combined_data.shape


# Solution: We get 103 rows as expected when we do a inner join, so we have good data.

# ### Question 3 ###

# Drawing inspiration from the previous lab where we cluster the neighbourhood of NYC, We cluster Toronto based on the similarities of the venues categories using Kmeans clustering and Foursquare API.

# In[22]:


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from sklearn.cluster import KMeans
get_ipython().system('pip install folium')
get_ipython().system('pip install geopy')
from geopy.geocoders import Nominatim 
import matplotlib.cm as cm
import matplotlib.colors as colors


print('Libraries imported.')


# In[23]:


address = 'Toronto, Ontario'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The coordinates of Toronto are {}, {}.'.format(latitude, longitude))


# Let's visualize the map of Toronto

# In[24]:


import folium
# Creating the map of Toronto
map_Toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# adding markers to map
for latitude, longitude, borough, neighbourhood in zip(combined_data['Latitude'], combined_data['Longitude'], combined_data['Borough'], combined_data['Neighbourhood']):
    label = '{}, {}'.format(neighbourhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [latitude, longitude],
        radius=5,
        popup=label,
        color='red',
        fill=True
        ).add_to(map_Toronto)  
    
map_Toronto


# Initializing Foursquare API credentials

# In[25]:


CLIENT_ID ='D5NS0RWF0XDLPNIVH3G3FTPERERVZQ35DE3XTZIBEDN5VQQA'
CLIENT_SECRET = 'C4PKNTCA0RD1F2M4TTBZ1IUHOJZKJL303GOHHCDU3M1IVZ2K'
VERSION = '20180604' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# Next, we create a function to get all the venue categories in Toronto

# In[26]:


def getNearbyVenues(names, latitudes, longitudes):
    radius=500
    
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius
            )
            
        # make the GET request
        results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Category']
    
    return(nearby_venues)


# Collecting the venues in Toronto for each Neighbourhood

# In[27]:


venues_in_toronto = getNearbyVenues(combined_data['Neighbourhood'], combined_data['Latitude'], combined_data['Longitude'])


# In[28]:


venues_in_toronto.shape


# 
# So we have 1317 records and 5 columns. Checking sample data

# In[29]:



venues_in_toronto.head()


# 
# Checking the Venues based on Neighbourhood

# In[30]:


venues_in_toronto.groupby('Neighbourhood').head()


# So there are 405 records for each neighbourhood.
# 
# Checking for the maximum venue categories

# In[31]:


venues_in_toronto.groupby('Venue Category').max()


# There are around 232 different types of Venue Categories. Interesting

# ## One Hot encoding the venue Categories ##

# In[32]:


toronto_venue_cat = pd.get_dummies(venues_in_toronto[['Venue Category']], prefix="", prefix_sep="")
toronto_venue_cat


# Adding the neighbourhood to the encoded dataframe

# In[33]:


toronto_venue_cat['Neighbourhood'] = venues_in_toronto['Neighbourhood'] 

# moving neighborhood column to the first column
fixed_columns = [toronto_venue_cat.columns[-1]] + list(toronto_venue_cat.columns[:-1])
toronto_venue_cat = toronto_venue_cat[fixed_columns]

toronto_venue_cat.head()


# We will group the Neighbourhoods, calculate the mean venue categories in each Neighbourhood

# In[34]:


toronto_grouped = toronto_venue_cat.groupby('Neighbourhood').mean().reset_index()
toronto_grouped.head()


# Let's make a function to get the top most common venue categories

# In[35]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[36]:


import numpy as np


# 
# There are way too many venue categories, we can take the top 10 to cluster the neighbourhoods

# In[37]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# 
# Let's make the model to cluster our Neighbourhoods

# In[38]:


# import k-means from clustering stage
from sklearn.cluster import KMeans


# In[39]:


# set number of clusters
k_num_clusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=k_num_clusters, random_state=0).fit(toronto_grouped_clustering)
kmeans


# Checking the labelling of our model

# In[40]:


kmeans.labels_[0:100]


# Let's add the clustering Label column to the top 10 common venue categories

# In[41]:



neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# Join toronto_grouped with combined_data on neighbourhood to add latitude & longitude for each neighborhood to prepare it for plotting

# In[42]:


toronto_merged = combined_data

toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head()


# Drop all the NaN values to prevent data skew

# In[43]:


toronto_merged_nonan = toronto_merged.dropna(subset=['Cluster Labels'])


# Plotting the clusters on the map

# In[44]:


import matplotlib.cm as cm
import matplotlib.colors as colors


# In[45]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(k_num_clusters)
ys = [i + x + (i*x)**2 for i in range(k_num_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged_nonan['Latitude'], toronto_merged_nonan['Longitude'], toronto_merged_nonan['Neighbourhood'], toronto_merged_nonan['Cluster Labels']):
    label = folium.Popup('Cluster ' + str(int(cluster) +1) + '\n' + str(poi) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster-1)],
        fill=True,
        fill_color=rainbow[int(cluster-1)]
        ).add_to(map_clusters)
        
map_clusters


# Let's verify each of our clusters
# 
# Cluster 1

# In[46]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 0, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# Cluster 2

# In[47]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 1, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# Cluster 3

# In[48]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 2, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# 
# Cluster 4

# In[49]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 3, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# 
# Cluster 5

# In[50]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 4, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# We have successfully cluster Toronto neighbourhood based on venue categories!

# In[ ]:




