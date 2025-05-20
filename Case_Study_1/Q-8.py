#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re

# Load the CSV file
df = pd.read_csv("Data.csv")  # Adjust path if necessary

# Function to count hashtags using regex
def count_hashtags(text):
    if pd.isnull(text):
        return 0
    return len(re.findall(r"#\w+", text))

# Apply the function to the Data column and sum the results
total_hashtags = df['Data'].apply(count_hashtags).sum()

# Display the total number of hashtags
print("Total number of hashtags:", total_hashtags)

