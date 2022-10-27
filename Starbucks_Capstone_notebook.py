#!/usr/bin/env python
# coding: utf-8

# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="pic1.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="pic2.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# In[1]:


#pip install seaborn==0.9.1


# In[2]:


import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import date
from scipy import stats

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# ## 1. Data Cleaning

# ### Portfolio
# 
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)

# In[3]:


portfolio.head()


# In[4]:


portfolio.info()


# In[5]:


portfolio.isna().sum()


# In[6]:


portfolio[['email','mobile','social','web']] = pd.get_dummies(portfolio.channels.apply(pd.Series).stack()).sum(level=0)


# In[7]:


portfolio = portfolio.drop(columns='channels')


# In[8]:


portfolio = portfolio.rename(columns={'id':'offer_id'})


# In[9]:


portfolio.duplicated().sum()


# In[10]:


portfolio.head()


# In[11]:


portfolio.describe()


# ### Profile
# * age (int) - age of the customer
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income

# In[12]:


profile.head()


# In[13]:


profile.info()


# In[14]:


profile.duplicated().sum()


# In[15]:


profile.isna().sum()  


# In[16]:


profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')


# In[17]:


profile = profile.rename(columns={'id':'customer_id'})


# In[18]:


profile['today_date'] = pd.to_datetime(date.today().strftime('%Y-%m-%d'))


# In[19]:


profile['days_member'] = (profile['today_date'] - profile['became_member_on']).dt.days


# In[20]:


profile.dtypes


# In[21]:


profile.describe()


# In[22]:


profile.loc[profile['age'] == 118].count()


# In[23]:


#We'll drop all the columns that the customers have age = 118, gender 
#and income NaN
profile = profile.dropna(subset=['income'])


# In[24]:


profile.head()


# In[25]:


profile.shape


# In[26]:


profile = profile.drop(columns='today_date')


# ### Transcript
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record

# In[27]:


transcript.head()


# In[28]:


transcript.info()


# In[29]:


def clean_dict(val):
    if list(val.keys())[0] in ['offer id', 'offer_id']:
        return list(val.values())[0]


# In[30]:


transcript['offer_id'] = transcript.value.apply(clean_dict)


# In[31]:


transcript = transcript.drop(columns='value')


# In[32]:


#we found some duplicates, but we'll not drop it because a person can receive
# an offer multiple times 
transcript.duplicated().sum()


# In[33]:


transcript.isna().sum() / transcript.isna().count() 


# In[34]:


transcript = transcript.rename(columns={'person':'customer_id'})


# In[35]:


transcript['event'].unique()


# In[36]:


transcript[['offer received', 'offer viewed', 'transaction', 'offer completed']] = pd.get_dummies(transcript['event'])


# In[37]:


transcript = transcript.drop(columns='event')


# In[38]:


transcript.head()


# ### Joining all

# In[39]:


profile.shape


# In[40]:


transcript.shape


# In[41]:


df = transcript.merge(profile,
                     on='customer_id',
                     how='left')\
              .merge(portfolio,
                     on='offer_id',
                     how='left')   


# In[42]:


df['age_bins'] = pd.cut(x=df["age"], bins=[18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
df['income_bins'] = pd.cut(x=df["income"], bins=[30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000,  120000])


# In[43]:


df.head()


# In[44]:


#keep in mind that we have many lines for the same customer_id 
df.loc[df['customer_id'] == '78afa995795e4d85b5d9ceeca43f5fef']


# In[66]:


#df that we have just one customer per line 
df_consolidated = transcript.groupby('customer_id').agg({'offer received':'sum',
                                                       'offer viewed':'sum',
                                                       'offer completed':'sum'
                                                      }).reset_index()\
                                                        .merge(profile,
                                                                on='customer_id')


# In[69]:


df_consolidated.shape


# In[76]:


d=df_consolidated.merge(df[['customer_id', 'income_bins', 'age_bins']],
                      how='inner',
                      on='customer_id')


# In[77]:


d.shape


# ## 2. Data Analysis

# ### 1. Is there a correlation between income and amount of completed offers?

# In[47]:


sns.heatmap(df_consolidated.corr(), 
            annot=True)


# P valores:

# In[56]:


stats.pearsonr(df_consolidated['days_member'], df_consolidated['offer completed'])


# In[57]:


stats.pearsonr(df_consolidated['income'], df_consolidated['offer completed'])


# In[58]:


stats.pearsonr(df_consolidated['age'], df_consolidated['offer completed'])


# The amount of offers completed shows correlation with:
# * days_member (0.43)
# * income(-0.27)
# * age(-0.16)
# 
# Obs: all these correlations have p value < 5%

# ### 2. What is the proportion of clients who have completed offers based on their age?

# In[85]:


completed = df.loc[df['offer completed'] == 1]


# In[106]:


p = completed.groupby('age_bins')['customer_id'].count()/ completed.shape[0] * 100


# In[107]:


p


# In[86]:


plt.figure(figsize=(7,5))
sns.countplot(completed['age_bins'],
              color='#036B52')
sns.despine()


# In[110]:


sns.boxplot(y='age',
            x='offer completed',  
            data=df,
           color='#036B52')


# 20% of the transactions that completed offers were clients with ages between 50 and 60 years old.
# 
# 50% of the transactions that completed offers were clients between 40 and 70 years old.

# ### 3. What is the proportion of clients who have completed offers based on their income?

# In[93]:


plt.figure(figsize=(15,5))
sns.countplot(completed['income_bins'],
              color='#036B52')
sns.despine()


# In[108]:


m = completed.groupby('income_bins')['customer_id'].count()/ completed.shape[0] * 100


# In[109]:


m


# 17% of the transactions that completed offers were clients with income between 50000 and 60000. In addition, 45% of the completed offers were from clientes with income less than 60000.

# In[98]:


sns.boxplot(y='income',
            x='offer completed',  
            data=df,
           color='#036B52')


# In[ ]:




