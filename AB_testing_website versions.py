#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.stats.api as sms
from math import ceil


# ## In this project, we need to compare the old version of a website to the new version. 
#     1. We need to analyze which version of the website is more preferred by users.
#     2. We need to see if our new website can obtain a conversion rate of 15%. Then it can be launched to all the users.
#     

# In[8]:


effect_size = sms.proportion_effectsize(0.13,0.15)
required_n = sms.NormalIndPower().solve_power(effect_size,power = 0.8, alpha = 0.05,ratio = 1)
required_n = ceil(required_n)
print(required_n)


# In[5]:


version = pd.read_csv("C:/Users/asus/Desktop/Website_Version/ab_data.csv")


# In[10]:


version.head()


# In[13]:


version.shape


# In[11]:


version.info()


# 1. We have a group feature where 'control' is the older version of the website and 'terminal' is the new version
# 2. The feature 'converted' is a binary feature. If it is '1', the user preferred the other version of the website. If it's '0', the user selected the same version of the website

# In[12]:


pd.crosstab(version['group'], version['landing_page'])


# In[14]:


version['user_id'].value_counts


# # We will check if there are any duplicate 'user_id' present.

# In[17]:


session_counts = version['user_id'].value_counts(ascending = False)
multiple_users = session_counts[session_counts > 1].count()
print('{} users appear multiple times'.format(multiple_users))


# In[18]:


drop_users = session_counts[session_counts > 1].index


# In[19]:


drop_users


# In[21]:


version = version[~version['user_id'].isin(drop_users)]
print(f'updated dataset has {version.shape[0]} entries')


# 1. We only have 3894 duplicate users out of 294478. So it seems to be a small number. Let's go ahead and delete the rows. 

# ## Taking a Sample
# 1. We are not considering the whole data. Instead, we take a sample to see the user's behavior. 4720 random user_id's from our dataset for each of the 'group'
# 

# In[23]:


control_sample = version[version['group'] == 'control'].sample(n = required_n, random_state = 22)
treatment_sample = version[version['group'] == 'treatment'].sample(n = required_n, random_state = 22)


# In[24]:


ab_test = pd.concat([control_sample,treatment_sample],axis  = 0)
ab_test.reset_index(drop = True, inplace = True)

ab_test


# In[25]:


ab_test.info()


# In[26]:


ab_test['group'].value_counts()


# ## Let's have a look at some basic statistics. 

# In[40]:


conversion_rates = ab_test.groupby('group')['converted']

std_p = lambda x: np.std(x, ddof=0)              # Std. deviation of the proportion
se_p = lambda x: stats.sem(x, ddof=0)            # Std. error of the proportion (std / sqrt(n))

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rates.style.format('{:.3f}')


# 1. Our new design of the website is performing slightly better than our old version.

# In[42]:


import seaborn as sns
sns.barplot(ab_test['group'], ab_test['converted'] , ci = False)
plt.title('Conversion rate by group')
plt.xlabel('Group')
plt.ylabel('Converted')


# In[47]:


control_results.count()


# ## Performing Hypothesis Testing
# 1. If our p-value is less than alpha(0.05), then we can reject the null hypothesis and say users found new website design more appealing.
# 2. If we fail to reject the null hypothesis, it means that the older version is preferred by most of the users.

# In[46]:


from statsmodels.stats.proportion import proportions_ztest, proportion_confint

control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']

n_con = control_results.count()
n_treat = treatment_results.count()
success = [control_results.sum(), treatment_results.sum()]
nobs = [n_con, n_treat]


# In[57]:


z_stat, pval = proportions_ztest(success, nobs = nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(success, nobs=nobs, alpha = 0.05)

print(f'z-statistic : {z_stat}')
print(f'p_value: {pval}')
print(f'ci 95% for control group: [{lower_con}], [{upper_con}]')
print(f'ci 95% for treatment group:[{lower_treat}, [{upper_treat}]')


# ## Conclusion
# 1. We can see the p-value is 0.73, which is greater than our alpha(0.05).
# 2. We cannot reject our null hypothesis in this case.
# 3. It means that the older version is preferred by most of the users than the new version
# 4. We can see the treatment group conversion rate is 13.5%. We need to achieve a 15% conversion rate to consider success in developing a new version of the website

# In[ ]:




