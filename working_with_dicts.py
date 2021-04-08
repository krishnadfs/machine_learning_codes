# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:11:20 2021

@author: MADHAVI
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import sys 

# https://towardsdatascience.com/working-with-python-dictionaries-a-cheat-sheet-706c14d29da5

# In[0.1]: keys and respective key values will be mapped in dicts

grocery_items = {"eggs": 4.99,
                 "banana": 1.49,
                 "cheese": 4.5,
                 "eggplant": 2.5,
                 "bread": 3.99}

print('keys in the data dict - Grocery Items:', grocery_items.keys())
print('recent item in the data dict - Grocery items:', grocery_items.popitem())

for keys_ in grocery_items.keys():
    print ('name of the key', keys_, 'and value of the key', grocery_items[keys_])
print('number of keys in the new dict:', len(grocery_items.keys()))
 
# In[0.2]: adding a new key into an existing dict
grocery_items["onion"] = 3.50 
print('number of keys in the new dict:', len(grocery_items.keys()))
print('recent item in the data dict - Grocery items:', grocery_items.popitem())

# In[0.3]: adding a list of entries for a specific key 
grocery_items["eggs"] = [3.99, 4.99, 5.50]

# In[0.4]: list of grocery items 
list_of_items = list(grocery_items)
print('list of grocery items ...', list_of_items)

# In[0.5]: checking the availibility of required item 
print('checking the availability of POTATOs in grocery items', "potato" in grocery_items)

# In[0.6]: printing individual items of interest from the dict 
print('identifying the items of interest ...\n')
for items_ in grocery_items:
    if "e" in items_:
        print('items ', items_)

# In[0.7]: 
list_of_items = [x for x in grocery_items.items()]
print('list of items and corresponding information:', list_of_items)

values_of_items = [x for x in grocery_items.values()]
print('values of individual items in the list:', values_of_items)