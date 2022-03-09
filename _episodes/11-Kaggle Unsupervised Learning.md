---
title: "Kaggle online competition: Unsupervised Learning"
teaching: 20
exercises: 0
questions:
- "How to participate in a Kaggle online compeition"
objectives:
- "Download Kaggle data and apply some algorithm technique that you have learnt to solve the actual data"
keypoints:
- "Kaggle"
---

# Mini Project: Unsupervised Learning with Kaggle Heights dataset

In previous chapter, you have worked with Supervised Learning data, now in this chapter, let's confront with another type of ML problem, which is Unsupervised Learning

https://www.kaggle.com/majyhain/height-of-male-and-female-by-country-2022

![image](https://user-images.githubusercontent.com/43855029/156072300-db4c4630-6653-4fea-9fed-76925011b855.png)

_**Project description:**_
The metric system is used in most nations to measure height.Despite the fact that the metric system is the most widely used measurement method, we will offer average heights in both metric and imperial units for each country.To be clear, the imperial system utilises feet and inches to measure height, whereas the metric system uses metres and centimetres.Although switching between these measurement units is not difficult, countries tend to choose one over the other in order to maintain uniformity.


For simpilicity: I downloaded the data for you and put the data table here:
https://raw.githubusercontent.com/vuminhtue/SMU_Machine_Learning_Python/master/data/Height%20of%20Male%20and%20Female%20by%20Country%202022.csv

## 11.1 Understand the data


There is only 1 csv file: Height of Male and Female by Country 2022

The dataset contains six columns:
• Rank
• Country Name
• Male height in Cm
• Female height in Cm
• Male height in Ft
• Female height in Ft


**Objective:**
- We will use Unsupervised ML to classify the groups of countries having similar heights of male and female
- Visualize the output


## Step 1: Load data from Kaggle Heights dataset

```python
import pandas as pd
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/vuminhtue/SMU_Machine_Learning_Python/master/data/Height%20of%20Male%20and%20Female%20by%20Country%202022.csv")
df.head()

```

Select only Male and Female Height in Ft

```python
X = df[["Male Height in Ft","Female Height in Ft"]]
```

## Step 2: Find the optimal K values:

- Using WSS method:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
wss = []
for k in range(1,10):
    model = KMeans(n_clusters=k).fit(X)
    wss.append(model.inertia_)
    
plt.scatter(range(1,10),wss)
plt.plot(range(1,10),wss)
plt.xlabel("Number of Clusters k")
plt.ylabel("Within Sum of Square")
plt.title("Optimal number of clusters based on WSS Method")
plt.show()
```

## Step 3: Apply K-Means and plot the clusters:

```python
model_KMeans = KMeans(n_clusters=3)
model_KMeans.fit(X)
centers = model_KMeans.cluster_centers_
df[df["Country Name"]=="United States"]["Male Height in Ft"]
USx = df[df["Country Name"]=="United States"]["Male Height in Ft"]
USy = df[df["Country Name"]=="United States"]["Female Height in Ft"]

plt.scatter(X["Male Height in Ft"],X["Female Height in Ft"],c=model_KMeans.labels_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.scatter(USx, USy, c='red', s=200, alpha=0.5);

plt.xlabel("Male Height in Ft")
plt.ylabel("Female Height in Ft")
plt.title('KMeans clustering with 3 clusters')
plt.show()

```

Show the country in cluster 1:

```python
df["Clusters"] = model_KMeans.labels_
df[df["Clusters"]==1]
```
