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

## Noisy Moon

- To create noisy moon, we use the datasets make_moons:

```python
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
X,y = noisy_moons
```

- The sensitivity of noisy moon with different perplexity is plotted below:

![Noisy_moon](https://github.com/user-attachments/assets/04485d3d-5fae-4861-a467-1b3247c638a4)

- Similarly, we have the animation of TSNE with 2D for this Noisy moon:

![noisy_moon](https://github.com/user-attachments/assets/0fa0a2dd-821a-4329-bab8-1980b67eaf56)

## Blobs data with 3 groups

- Next, Let's create data with 3 separate group from make_blobs function in datasets

```python
X, y = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],random_state=123)
```

- We can visualize the Raw data with TSNE from different perplexity as below on 2D plane:

![Blobs](https://github.com/user-attachments/assets/5306146d-0c2f-4c80-b024-c37e05c0dec4)

- And the animation of TSNE visualization with perplexity range from 1-100:

![blobs](https://github.com/user-attachments/assets/7f6ff69e-25ea-4b44-b835-f2a6bb3cfd8d)

## MNIST data
- Not only using the 2D plane model, we can use TSNE to visualize the multi-dimension.
- One of the example that we are using is the digital number data called [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)
- To download the MNIST data, we can use tensorflow keras datasets:

```python
from tensorflow.keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
```

- The training data contains 60000 images and the testing contains 10000 images.
- The image has resolution of 28x28 and is in gray scale with color range from 0-255
- To quickly visualize the MNIST data:

```python
for i in range(64):
    ax = plt.subplot(8, 8, i+1)
    ax.axis('off')
    plt.imshow(X_train[i], cmap='Greys')
```

<img width="434" alt="image" src="https://github.com/user-attachments/assets/c586f5ac-19fd-4ffb-b962-3bd6e17d0a95">

- Because of its resolution 28x28 = 784, this dataset has the dimension of 784 and we can utilize TSNE to visualize the 10000 testing images in 2D or 3D scale
- First, let reshape the data:

```python
X = X_test.reshape(10000,28*28)
y = y_test
```

- And apply TSNE with 2D plane and default perplexity

```python
model_tsne = TSNE(n_components=2)
tsne_X = model_tsne.fit_transform(X)
df_tsne = pd.DataFrame(tsne_X,columns=['TSNE1','TSNE2'])
df_tsne['Cluster'] = y
```

- We can then visualize the MNIST dataset on 2D plane of TSNE:

```python
tsne = sns.lmplot(x="TSNE1", y="TSNE2",
  data=df_tsne, 
  fit_reg=False, 
  hue='Cluster', # color by cluster
  legend=True,
  scatter_kws={"s": 5}, # specify the point size
  height=8)
```

<img width="746" alt="image" src="https://github.com/user-attachments/assets/2905b525-4174-4069-a170-820e8a905d18">


- We can even visualize the 784 dimension data in 3D perspective by fitting TSNE with 3 dimensions:

```python
model_tsne = TSNE(n_components=3)
tsne_X = model_tsne.fit_transform(X)
df_tsne = pd.DataFrame(tsne_X,columns=['TSNE1','TSNE2','TSNE3'])
df_tsne['Cluster'] = y
```

```python
fig = plt.figure(figsize=(50, 30))
ax = plt.axes(projection ='3d')
# defining all 3 axis
x = df_tsne["TSNE1"]
y = df_tsne["TSNE2"]
z = df_tsne["TSNE3"]

# plotting
ax.scatter(x, y, z, c = df_tsne["Cluster"])
ax.set_title('3D line plot geeks for geeks')
plt.show()
```

<img width="977" alt="image" src="https://github.com/user-attachments/assets/7b0852a9-022e-41fa-8418-b0f404f24a6c">
