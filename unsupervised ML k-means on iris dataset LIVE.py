#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# In[3]:


from sklearn.datasets import load_iris
iris_data = load_iris()


# In[4]:


print(iris_data.DESCR)


# In[5]:


df = pd.DataFrame(iris_data.data)
df.columns = iris_data.feature_names
df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


sns.pairplot(df, markers='+')
plt.show()


# In[11]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[12]:


df.plot(kind='scatter', x="sepal length (cm)", y="sepal width (cm)")
plt.show()


# In[13]:


wss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df)
    wss.append(kmeans.inertia_)
    
plt.plot(range(1,10),wss)
plt.title('The elbow method')
plt.xlabel('Number of clsuters')
plt.ylabel('Sum of squared distances')
plt.show()


# In[14]:


for i in range(2,10):
    kmeans = KMeans(n_clusters=i, max_iter=100)
    kmeans.fit(df)
    score = silhouette_score(df, kmeans.labels_)
    print("For cluster: {}, the silhouette score is: {}".format(i,score))


# In[15]:


silouette_coeff = []

for i in range(2,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    score = silhouette_score(df, kmeans.labels_)
    silouette_coeff.append(score)
    
plt.plot(range(2,10), silouette_coeff)
plt.xticks(range(2,10))
plt.xlabel("Number of clusters")
plt.ylabel("silhouette coeff")
plt.show()


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()
X = scaler.fit_transform(df)


# In[18]:


kmeans = KMeans(n_clusters=3)
y = kmeans.fit_predict(X)
unique_labels = np.unique(y)


# In[19]:


plt.figure(figsize=(10,5))
for i in unique_labels:
    plt.scatter(X[y==i, 0], X[y==i, 1], label=i)
plt.title("Iris data clusters")
plt.legend()
plt.show()

