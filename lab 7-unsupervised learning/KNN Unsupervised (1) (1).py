#!/usr/bin/env python
# coding: utf-8

# In[500]:


import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score,confusion_matrix
import random
from statistics import mean
# import statsmodels.api as sm
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
from sklearn.cluster import KMeans


# In[501]:


np.random.seed(40)


# In[502]:


df = pd.read_csv("fruit_data_with_colors.txt",delimiter = "\t")


# In[ ]:





# In[503]:


df=df.drop('fruit_label', axis=1)


# In[ ]:





# In[504]:


df1=df.sample(n=4)


# In[505]:


df1=df1.drop(['fruit_name','fruit_subtype'],axis=1)


# In[506]:


X = df[["mass","width","height","color_score"]]


# In[507]:


# plt.scatter(df['width'],df['height'])
# plt.show()


# In[ ]:





# In[508]:


def euclidian_distance(query,X):
        difference = np.array(X) - np.array(query)
        sqrd_diff = np.square(difference)
        sum_sqrd_diff = np.sum(sqrd_diff, axis = 1)
        distance = np.sqrt(sum_sqrd_diff)
        return distance


# In[509]:


c=np.array(df1)
c


# In[510]:


xd=np.array(X)


# In[ ]:





# In[511]:


euclidian_distance(xd[0],c)


# In[ ]:





# In[512]:


# c1=[]
# c2=[]
# c3=[]
# c4=[]


# In[513]:

print(c)
for j in range(10):
    # print('iter')
    c1=[]
    c2=[]
    c3=[]
    c4=[]

    for i in range(len(xd)):
        index=np.argmin(euclidian_distance(xd[i],c))
        print(index)
        if index==0:
            c1.append(xd[i])
        elif index==1:
            c2.append(xd[i])
        elif index==2:
            c3.append(xd[i])
        else:
            c4.append(xd[i])
            
    # print(c3)
    # print(c1)
    # print(c1,'\n',c2,'\n',c3,'\n',c4)
        
    c[0]=np.array(np.mean(c1,axis=0))
    c[1]=np.array(np.mean(c2,axis=0))
    c[2]=np.array(np.mean(c3,axis=0))
    c[3]=np.array(np.mean(c4,axis=0))
    


# In[514]:


c1,c2,c3,c4


# In[515]:


# a=np.mean(c1,axis=0)
# b=np.mean(c2,axis=0)
# c=np.mean(c3,axis=0)
# d=np.mean(c4,axis=0)


# In[516]:



# In[ ]:





# In[ ]:





# In[ ]:




