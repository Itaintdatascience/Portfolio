
# coding: utf-8

# # Principal Components Analysis with Wine Data

# In[1]:


#Imports
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.datasets import load_digits

plt.style.use('fivethirtyeight')


# In[2]:


#Load in the wine data
wine = pd.read_csv("../data/winequality_merged.csv")
wine.head()


# In[3]:


wine.shape


# In[5]:


wine.red_wine.value_counts(normalize=True)
# ~25% of the responses are red wine.


# Which attributes here can we use to identify which class of wine it belongs to?

# In[6]:


#Assign red_wine column to variable rw 
rw = wine.red_wine

#Drop red_wine from wine

wine.drop("red_wine", axis = 1, inplace= True)


# First let's determine the correlations between the independent variables

# In[16]:


#Correlation heatmap

sb.heatmap(wine.corr(), annot=True, annot_kws={"size": 9})


# Multiculineartiy exists when columns highly correlated with one another. One benefit of PCA is that it can deal with multicollinearity pretty well because multicollinearity simply means that you have excess dimensions in your data.

# In[17]:


#We need to standardize or scale the data before we can transform our data because the measurements vary 
# among the variables.

#Intialize scaler, you can use minmaxscaler as well.. both the same.
scale = StandardScaler()

#Fit and transform wine data using standard scaler
wine_s = scale.fit_transform(wine)

#let's take a look at the first row of data by the first slice:
wine_s[0]


# In[18]:


#Intialize PCA object
#We're deliberating leaving the n_components parameters alone
pca = PCA()

#Fit and transform wine_s use pca
wine_pca = pca.fit_transform(wine_s)

#Number of components
pca.n_components_


# In[20]:


pca.n_features_


# In[19]:


pca.explained_variance_


# We need to reduce the number of dimensions from 12, but how do we select how many to keep? Let's visualize how much
# variance is explained for each principal component. They will be ranked by the greatest explained variance first. A cumulative variance of 85% can be a great rule of thumb for our threshold in selecting number of components. 
# 

# In[21]:


#Shows the percentage of the variance explained by each component
pca.explained_variance_ratio_


# In[23]:


plt.figure(figsize=(10, 8))

components = range(1, pca.n_components_ +1)

plt.bar(components, pca.explained_variance_ratio_, label = "Explained Variance Ratio")
plt.plot(components, np.cumsum(pca.explained_variance_ratio_), 
         c = "r", label = "Cumulative Sum of Explained Variance ratios")

plt.xlabel("Components")
plt.ylabel("Explained Variance")
plt.legend();


# At the 85% explained variance level, we are at about 7 coponents. At just 2 components, you can get about 50% of the explained variance (essense of the original dataset). This tell us that if we view this 12-dimension data on a 2D scatter plot then we would be seeing about half of the total variance in the dataset.

# In[48]:


pca.components_[1].mean()


# In[28]:


pca.explained_variance_.sum()


# In[56]:


pca.explained_variance_ratio_[:3].sum()


# How do you label the components based on the weights?
# 
# Let's print out the component weights with their corresponding variables for PC1, PC2, and PC3

# In[31]:


#PC1
for col, comp in zip(wine.columns, pca.components_[0]):
    print col, comp

    #Total sulfur dioxide has the strongest weight


# In[32]:


#PC2
for col, comp in zip(wine.columns, pca.components_[1]):
    print col, comp
    
    #density 


# In[33]:


#Component 3
for col, comp in zip(wine.columns, pca.components_[2]):
    print col, comp


# In[44]:


#Create color values from rw
colors = rw.map({0:"r", 1:"b"})

plt.figure(figsize=(8,7))
plt.scatter(wine_pca[:, 0], wine_pca[:, 1], c=colors, alpha=.4);


# 48% of the cumulative variance of this dataset is plotted. We can make a pretty decent model by putting a line down the middle here.
# 
# Here's the same graph in 3d: 
# 
# https://plot.ly/~DarrenK.Lee/1/

# # Logistic Regression Model Selection

# In[82]:


X = wine_pca[:,:2]
y = rw
# We previously saved the variable rw as the target column, now we will try to build a model to predict if 
# the wine is a red bottle or not. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)


# In[68]:


y.value_counts(normalize= True)

# remember that our null accuracy score to beat is 75%. We can assume that the entire dataset is not red wine,
# and we would be about 75% accurate about that statement. Let's beat this baseline number!


# In[65]:


lr = LogisticRegression()
lr.fit(X_train, y_train)

lr.score(X_train, y_train)


# In[66]:


lr.score(X_test, y_test)


# In[69]:


preds = lr.predict(X_test)


# In[73]:


cm = confusion_matrix(y_test, preds)
c_matrix = pd.DataFrame(cm, columns= ("Predicted Not Red Wine", "Predicted Red Wine"))
c_matrix


# The trained model was applied to new/unseen data in the test set. We see a 95.6% accuracy with the test set in classification. Not bad for a model without a specific boost or optimized parameter tuning! Below, you can see the cross validated scores if we were to include all of the components into play. This is a matter of the context of your data. Falsely classifying may be more serious when you're dealing with Credit Card fraud over classifying wine types.

# In[67]:


components = range(1, 13)
scores = []

for i in components:
    pca = PCA(n_components=i)
    pca_data = pca.fit_transform(wine_s)
    score = cross_val_score(LogisticRegression(), pca_data, rw, cv = 5, scoring = "accuracy").mean()
    scores.append(score)
    
plt.plot(components, scores)
plt.xlabel("N Components")
plt.ylabel("CV Accuracy Score"); 

