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
- "perplexity" can be considered to be number of nearest neighbors (the higher the number, the better fit the data) when considered for each data point when constructing the probability distribution of similarities in high-dimensional space.
- According to authors, "The performance of SNE is fairly robust to changes in the perplexity, and typical values are between 5 and 50"
- This chapter discusses some aspect of TSNE and its perplexity with different kind of input data. The code will be given as well.
- What is good perplexity?
  
      + typical range 5-50: small data (5-30), large data (30-50)
      + 1-5: disconnected structure and poorly representation
      + > 50: overlapping structure and loss detail
  

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

- Animation for perplexity from 1 to 50:

![MNIST](https://github.com/user-attachments/assets/00269907-577a-4690-a1df-95425c0c6f0f)

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

- Additionally, we can also plot using raw images from Xtest data:

```python
from PIL import Image

tx, ty = df_tsne['TSNE1'], df_tsne['TSNE2']
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

width = 4000
height = 3000
max_dim = 10
full_image = Image.new(mode="RGB", size=(width, height),color = (255, 255, 255))
for idx, x in enumerate(X_test):
    tile = Image.fromarray(np.uint8(x * 255))
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                            int((height-max_dim) * ty[idx])))
full_image
full_image.save("MNIST_number.jpg")
```

![MNIST_number](https://github.com/user-attachments/assets/b72c0746-67f5-4e23-8557-24ef2b85385e)


## FMNIST data
- Fashion MNIST is similar to MNIST data but using fashion image like shirt, trouser, shoes, etc 
- One of the example that we are using is the digital number data called [FMNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- To download the FMNIST data, we can use tensorflow keras datasets and we will visualize using testing data

```python
from tensorflow.keras.datasets import fashion_mnist
(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
X = X_test.reshape(10000,28*28)
y = y_test
```

- Setup the model:

```python
model_tsne = TSNE(n_components=2,perplexity=30)
tsne_X = model_tsne.fit_transform(X)

Cluster_name = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
                "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

df_tsne = pd.DataFrame(tsne_X,columns=['TSNE1','TSNE2'])
df_tsne['Cluster'] = y
df_tsne['Cluster_name'] = pd.Series(y).apply(lambda x:Cluster_name[x])
```

- Visualize with TSNE using default perplexity:

```python

tsne1 = sns.lmplot(x="TSNE1", y="TSNE2",
  data=df_tsne, 
  fit_reg=False, 
  hue='Cluster_name', # color by cluster
  legend=True,
  scatter_kws={"s": 5}, # specify the point size
  height=8)
```

![image](https://github.com/user-attachments/assets/b46bdd34-aef8-4bb2-90a3-2a56c7813cbc)


- We can also visulize using images and we can clearly the grouping to images by clicking on the image to zoom in :

```python
tx, ty = df_tsne['TSNE1'], df_tsne['TSNE2']
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

from PIL import Image
width = 4000
height = 3000
max_dim = 10
full_image = Image.new(mode="RGB", size=(width, height),color = (255, 255, 255))


for idx, x in enumerate(X_test):
    tile = Image.fromarray(np.uint8(x * 255))
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                            int((height-max_dim) * ty[idx])))

full_image.save("FMNIST.jpg")

```

![FMNIST](https://github.com/user-attachments/assets/c7b503cf-b781-4388-aedd-1a20c20375ee)



## Dashboard with Python Shiny:

[Dashboard](https://shinylive.io/py/app/#code=NobwRAdghgtgpmAXGKAHVA6VBPMAaMAYwHsIAXOcpMASxlWICcyACGKM1AG2LK5oBGWbN14soAZxbcyAHQh0GzFhACu9bOKkRU8xU1aooEACaSt0k3voGWEuFAFMIFiRAnyAZo2Iw7Aay4HRggMdgVPYi4TFn1lABUAZQA5AFEvHz8JQOCXONZCLlUJCkY8FjMySTgyCXKYGgAPMlVGOAzfAKCoEIxC4tLYm2UAaQBZB3cOrJye0JM4EhsJGjIaUiGlVgAFAGEAQWm7AAsaCGwMOEbUNokpfJZVGnKz1FUycrbTOEYjiVPzptbG0oIQ1gA3doQeTyADuq2OjxoGBWCwEPQAFABKRDyFj4pEYV7vAD69iCYIxsjAqH8Zys+BY1MScAprH+aDgHkZeIJfP5ApYIGpEGINAk2BJhBojEKXOpiCZYGSYolLF2Mrl1PKItVkpgxFI3MV1JV4s0Y0N0PwvMFdoJ1IEPAExqVACFndy8NTjOLiArqfsFBJ-TbaIxxQGwABJABK0cS1IAvrb7Xasd6XASnkSdKSJPwFowqZBtUrkuoBD8WMRPCxY8YTJ0GGdamWAAzlACM7d75QArL2samc8SyGTCz8S4Qy6bK9Xayx+iUfl6WJ2WD3ygAmYdZ-GjvPj0FrUgkgTvMikEsCMgQEn+eDGNfU3bHQ32FgVmBVxg1uu7EUK6MFIkR-uMkzcnufKHm844FjQRYli+yrzn+i7bBGEDSkYXDqr4DAQJQbaMhuW4sLuI7ImOJInus94XmQV4QDed4kqghBQLOYBvh+cBfmh-4sJhZw4VAeG7ARpDEVBVG5nBE6IVO1K6DyYDbD83BXKs2Adt2fYsAAzO20HZtRR60WC9Hnpe16OmxtREdxvHEJ+qCaUEjQ6eIpjCTwrBJGk1J7vICx1qK5pSpqQQSNiuLQneepRbKMUsAAvBUHDVLUYRQP4cDJXKsUwnIbESLAWkSGlY4YCxGYsJ4J5MGl7YYP25QRfYLUYO27UsIwja+GSVQUGlxklSFZAABp4JoGWdZK0opfKCUmJ4U3pZYGAACJZQAYgN8AYjNJBFDA7hpcA1JTV22rXdu1IALqTWtU1XTxQGlE9aW6QlbQtCEFTraFcDhUlBpGnFJULSSEPuJtlTZRIuX5bDVrFQlshlRVMXVUetXYh1qpwN1vWfINMDDRwJPjQlk0zXNKjg+jJWvZtqAmDt+2HXAx14Kd6gXe9N13WAU0PWAz2s+t72AQMPzfb9WP-a0LivSDdZOsQLpQwlWsugjWX2Dl7Co-rGNY1j97lfQuM1XV5TLqUw0mJdXY9TubXlK1-aPeTphDSU1Nje2E0lQzm3m9LG0ZRzXNVAdsC8ydUSC1Vwu3WG4tPS9Muvp9CuSz9JUq4D6vfHWvohrrWPm4bVTG8jpsFebVKY9jttcnjcEE-VA0B5TQejbTWP07NkeeiVZD9xIYHsKeLgZcAwCtQAbOUAC0a9+ywwBbxgAAs3sYAAHI9j3h7RwbEJtOgYE2ZB8yw0-PnPHD0bnMdbbtCc88dV9+n5qnc66drqZ29GLCWUtVp5w+vLX4RclbTxqKrIGU0NaxAjLFHEJUZTinrkjDAPAoAmBJHg7B4dNrkPvllEqjNqFVEYAAcxqNHdmnMf5QETkdFOZ0hbXVDlncB91RZTUMjnaOssC4IMesXP6KCy7Awrs-NwBUpoAE0MSvRwQlA0CwuAkkcvxDKgVUgYnvEsQiMk0rbjwO5RgWkvJkGwN3d4WBsSTSMSSL+ejWSGNURgTwqxDEzznlo9aud-FEXYfHLhf8vG8LTpdAA5KYrsyS8CpJSKkbcyToFYzWlEuAUj4HfVeiU4CT0S4KLVp4IpGC9j7G8Zo7R8U+S+IMY0zajTzFSikkRcgVUarYNMviPYUgModJJI0wJwSX7uDCdo1MhSOJQBiZw7hvNxmjKBuxTiFSvpF3KfnUpktUyl1qXsri0IIDwjIIiHMXAoDYGIKSAW50SSwgGqgDE8ITD3LSl2FgAB6CiOJ8SpjuQ85EnFGAmF1naakAB5CMTCzjiUylUakqY+QAAEvhFiwP5DE4kyBpUDCwRInEmLVm2P5YKOKCRhTsNS0oPydFpnxDcVsGIaq0npB4xlAoaB1j5XSUw2J0oZV1JFJaRUAz7k5fiNm80kpypitiIV9pPDQHgJtU0ep1TRXlGALVfJWQitiPjflEqsRSqVDDOGxozWChVUzSKTrNWKqVTqpO+rlSGstEabF3rBQWtFda8V8K7VpWlWAKOSAXUCjda3HZPrdXGPLIaj02tuRJpYOGq1PcbXRvtT6a+CqlV8jdVXYgXqq0El9XquNQY-QhqVYWsVAqY1xvIZWhtaCqFYPrQOptmbqRxgTCG-NhABhDVwjUCgm0QDthNPGoo7RGRdjXUwYwLCyzbjXUwtolBkz5rcMjCQrKPK8AxI0clYtwEsBcSIm0oa0yIzSq9HU777THFUCTZJctgIZJYAAYiXFEJgLABCaCdgrX9doF00rSrOko87xKLrgJmAdLAggsNMGleIjAANYhRDUDEaw+Ak08NSEAY6kz1igLCaQ-lyjJGXfbbEKYwCTUhQiQksLo1tMFNSCCz51TSLOEw9tgp8WUEJaIR+pKH3iamLxrV8m6KQkuJCcgvL8a3nvI+SCaamWg1Y7wB8T53AIs5Zarttqy2QDVca51iHXXrVvq55aFCPMCjHf6s0aoNS+dk2mTtkbu3OcdSzRN-n+Ruti5DMzaZAtxuCxaOL+bIvFqjZK2NSoE0iYbSmz0I6G3pazeaFgOaXThftLltxJaCtxtrf20rXmMq1oq1WqrgYK2moS01sgWB8s9qVH2+LuG3XkN6+mv1cbJ2JiG0qqZamJksA270+DIFXGjcIIKhL62JjPlmY-cpGdRFQOgUqtDV5KbIaXRlFda6nQAbLNupUu6ID7sZIepUx64CnvwIZXEYA2gMjwAfcH2BWQ8FhNqfs4P2AEaxfgVe4OnSgn8NqAA7ODt4DigjahPuDoHul8AAE5wf8CYccMg73N0pgSxelE17ie3vvccx9T1yguJ59nSWOGB2fu-UyBLfJ-0kxO5BIhjhWQSBJOUCDp1oOwaXNI-NfInsk3uxhoINKRcNvwwpojJG4BkeNpR1YQQ0q0bABtoS9GM1JnY4JRcu3NsgBqodrEPG7X5v4-cwTPRhNaupI0hreKCU-CJbelTkeDgMt-VpqyOm4B6cfjVIzVzUsVAs0pq5dm0wOai05wrMqJSFRih1qtSWfNFXm5y-rAaauhflatzlI2xvRcry5j1cWSv166+66vnr892lb5llgQb1M5f4BGvLfe43Fe10y0fqb1-4mn9myeXeIuL6Lc18bzn2vTdF6Pnrk-BSt9bSGaPYaj+OdLf3qbw+lWzeHTfgLGb-XLcfzTHzRWU4gykaWaXCSmh-yXDnUe0wxpWXVXSKw3U+x3X7j+x3CPRPWtBZyVTZyvQ4DZWJXvQ3BcS7GN3ryyi-TqVWR-Vw2lzSiA2kVA1Vygz-A112232kHgNGn1zgMNwoAoKVVN0I2I1I3I0fioztwdy6UXBdyTjdwEh-AXDrHGU43xmwR4yENwyVUaCeSrC4AfT2CfWwH0NZCMN2AlkD29UtkVShVDzhRL35GpFMUAJYHk2+EYHj2Uy4DJRcOyRTztDTwhDgF02IgMx7lzyMXz2ZSLyMScLtDL2XwrzjRhnVRNQ-05Qb1lTc2bzSz-wy0NQ71rwP0a2f3L1f1SOZmDQv06y-mS1s2gL5F3xqznzzWG3KOSMqOQNzTr0-033KyaMbQKOqzVDq3aI7U6JPxXyVHP0yI-Sv2vjyO1RGIGzbVKLtB7xawm2pHfy4K-3FGWKn1WJjHjBWxhAS0KSMU2gSRaQiRnVgPYh4MzVe2QI+y3TQL3U3UwMB2wLPVZ3cHZ0IJvUfnvX8LSBMIfVMQlm0I-SoKuNUToIHQYKYPgRYMgx4HYLgy10lwJF11Q0eN11hPtBENdjEMtwkJt2o3t3BNSGdwY3KHsUcW8hez5W42ChYDACTEeiAA)


