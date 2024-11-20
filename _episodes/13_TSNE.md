---
title: "Data visualization with TSNE"
teaching: 20
exercises: 0
questions:
- "How to apply TSNE for data visualization?"
objectives:
- "Apply TSNE from scikit-learn package"
keypoints:
- "Visualization"
---

# 13. TSNE for Visualization
- t-SNE (t-distributed Stochastic Neighbor Embedding) is a machine learning algorithm that visualizes high-dimensional data in two or three dimensions.
- It was developed by Laurens van der Maaten and Geoffrey Hinton in  2008
- The algorithm is non-linear. Its goal is to take a set of points in a high-dimensional space and find a faithful representation of those points in a lower-dimensional space in 2D or 3D plane
- TSNE has tunable parameters “perplexity,” which says (loosely) how to balance attention between local and global aspects the data.
- According to authors, "The performance of SNE is fairly robust to changes in the perplexity, and typical values are between 5 and 50"
- This chapter discusses some aspect of TSNE and its perplexity with different kind of input data. The code will be given as well

## Load scikit-learn library and create some sample data
- The data is taken from datasets, used in Unsupervised Learning chapter that I introduced earlier in Chapter 9

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn import cluster, datasets, mixture

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
seed = 30
```

## Noisy Circle data

- Let's create noisy circle data in 2D plane that have X variables containing the coordinate of the points and y variable which contains the labels of predefinded group:

```python
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
X,y = noisy_circles
```

- Let's plot the noisy circle raw plot:

```python
dfX = pd.DataFrame(X,columns=["X1","X2"])
dfX["Cluster"]=y

sns.scatterplot(x="X1", y="X2",
    data=dfX, 
    hue='Cluster', # color by cluster
    legend=True).set(title="Noisy Circle Raw plot")
```
<img width="584" alt="image" src="https://github.com/user-attachments/assets/66685e9e-f48a-41b6-8a6e-40e7ec5cdaad">

- Next, let's apply TSNE to this dataset and visualize in 2D plane:
- The perplexity is not specified so it perplexity=30 by default

```python
model_tsne = TSNE(n_components=2)
tsne_X = model_tsne.fit_transform(dfX)
df_tsne = pd.DataFrame(tsne_X,columns=['TSNE1','TSNE2'])
df_tsne['Cluster'] = y

sns.scatterplot(x="TSNE1", y="TSNE2",
    data=df_tsne, 
    hue='Cluster', # color by cluster
    legend=True).set(title="TSNE")
```

<img width="587" alt="image" src="https://github.com/user-attachments/assets/a3a5aee3-cd8a-4759-b8cf-9e68c933f97c">

- We can visualize the transformation of TSNE with different variable of perplexity
- Here, we compare the raw data with perplexity range from 1 to 200:

![my_plot](https://github.com/user-attachments/assets/a53eca14-71de-4c7a-9c9f-1858641d4c02)

- And here is the animation code of TSNE for noisy circle with perplexity range from 1 to 100:

```python
import matplotlib.animation as animation
prange = range(0,100)

def animate(p):
    plt.clf()  # Clear the previous frame
    model_tsne = TSNE(n_components=2,perplexity=p+1)
    tsne_X = model_tsne.fit_transform(dfX)
    df_tsne = pd.DataFrame(tsne_X,columns=['TSNE1','TSNE2'])
    df_tsne['Cluster'] = y
    
    sns.scatterplot(x="TSNE1", y="TSNE2",
        data=df_tsne, 
        hue='Cluster', # color by cluster
        legend=True).set(title="Perplexity = {}".format(p))

ani = animation.FuncAnimation(plt.gcf(), animate, frames=len(prange),interval=10)
ani.save('noisy_moon.gif', writer='pillow') 
```

![noisy_circle_i10](https://github.com/user-attachments/assets/62b02479-1483-48a8-8526-ec141f74c079)
