---
title: "Neural Network"
teaching: 20
exercises: 0
questions:
- "How to use Neural Network in Machine Learning model"
objectives:
- "Learn how to use ANN in ML model"
keypoints:
- "ANN"
---

# 8 Neural Network

## 8.1 The Neural Network of a brain

- Neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. 
- Neuron is a basic unit in a nervous system and is the most important component of the brain.
- In each Neuron, there is a cell body (node), dendrite (input signal) and axon (output signal to other neuron).
- If a Neuron received enough signal, it is then activated to decide whether or not it should transmitt the signal to other neuron or not.

![image](https://user-images.githubusercontent.com/43855029/114472746-da188c00-9bc0-11eb-913c-9dcd14f872ac.png)

## 8.2 Neural Network in Machine Learning:

![image](https://user-images.githubusercontent.com/43855029/114472756-dd137c80-9bc0-11eb-863d-7c4d054efa89.png)

## 8.3 Formulation of Neural Network:


![image](https://user-images.githubusercontent.com/43855029/114472776-e997d500-9bc0-11eb-9f70-450389c912df.png)

Here:
- x1,x2....xn are input variables. 
- w1,w2....wn are weights of respective inputs.
- b is the bias, which is summed with the weighted inputs to form the net inputs. 

In which: 
- Bias and weights are both adjustable parameters of the neuron.
- Parameters are adjusted using some learning rules. 
- The output of a neuron can range from -inf to +inf. As the neuron doesn’t know the boundary, so we need a mapping mechanism between the input and output of the neuron. This mechanism of mapping inputs to output is known as Activation Function.

**Activation functions:**

![image](https://user-images.githubusercontent.com/43855029/114575672-6752f380-9c48-11eb-8d53-c78d052cdf17.png)

## 8.4 Multi-Layer Perceptron (MLP)

**Multi-layer Perceptron (MLP)** is a supervised learning algorithm.
Given a set of features `X = x1, x2, ... xm`, and target `y`, MLP can learn a non-linear function approximator for either classification or regression.

Between the input and the output layer, there can be one or more non-linear layers, called hidden layers. Figure below shows a one hidden layer MLP with scalar output.

![image](https://user-images.githubusercontent.com/43855029/114472972-51e6b680-9bc1-11eb-9e78-90ec739844ee.png)

![image](https://user-images.githubusercontent.com/43855029/114575549-48546180-9c48-11eb-8c9c-c5eac3180df1.png)

**The advantages of Multi-layer Perceptron:**
- Capability to learn non-linear models.
- Capability to learn models in real-time (on-line learning) using partial_fit.

**The disadvantages of Multi-layer Perceptron:**
- MLP with hidden layers have a non-convex loss function where there exists more than one local minimum. Therefore different random weight initializations can lead to different validation accuracy.
- MLP requires tuning a number of hyperparameters such as the number of hidden neurons, layers, and iterations.
- MLP is sensitive to feature scaling.

## 8.5 Type of Neural Network Multi-Layer Perceptron in sklearn
Similar to previous Machine Learning model, there are 2 main types of MLP in sklearn, depending on the model output:
- MLPClassifier: for Classification problem
- MLPRegressor: for Regression problem 

## 8.6 Implementation with Classification problem

Here we use **Breast Cancer Wisconsine** data for Classification problem

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The Class **MLPClassifier** implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
There are lots of parameters in MLPClassifier:
- **hidden_layer_sizes** which is the number of hidden layers and neurons for each layer. Default=`(100,)`
for example `hidden_layer_sizes=(100,)` means there is 1 hidden layers used, with 100 neurons.
for example `hidden_layer_sizes=(50,20)` means there are 2 hidden layers used, the first layer has 50 neuron and the second has 20 neurons.
- **solver** `lbfgs, sgd, adam`. Default=`adam`
- **activation** `identity, logistic, tanh, relu`. Default='relu`

More information can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

```python
from sklearn.neural_network import MLPClassifier
model_NN = MLPClassifier(hidden_layer_sizes = (50,20),solver='lbfgs',activation='relu',random_state=123).fit(X_train_scaled, y_train)
model_NN.score(X_test_scaled,y_test)
```

## 8.7 Implementation with Regression problem
- Class **MLPRegressor** implements a multi-layer perceptron (MLP) that trains using backpropagation with no activation function in the output layer, which can also be seen as using the identity function as activation function. 
- Therefore, it uses the square error as the loss function, and the output is a set of continuous values.

Here we use **california housing** data from Regression espisode:

```python
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

# Predictors/Input:
X = pd.DataFrame(data.data,columns=data.feature_names)

# Predictand/output:
y = pd.DataFrame(data.target,columns=data.target_names)
```                                                    

Fit **MLPRegressor** model
```python
from sklearn.neural_network import MLPRegressor
model_NN = MLPRegressor(hidden_layer_sizes = (10,5),solver='lbfgs',activation='tanh',max_iter=1000).fit(X_train,y_train)
model_NN.score(X_test,y_test)
```

## 8.8 Tips on using MLP
- Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. 
- Empirically, we observed that **L-BFGS** converges faster and with better solutions on small datasets. For relatively large datasets, however, **Adam** is very robust. It usually converges quickly and gives pretty good performance. **SGD** with momentum or nesterov’s momentum, on the other hand, can perform better than those two algorithms if learning rate is correctly tuned.
- Since backpropagation has a high time complexity, it is advisable to start with smaller number of hidden neurons and few hidden layers for training.
- The loss function for Classifier is **Cross-Entropy** while for Regression is **Square-Error**

## 8.9. Notes
- There are many other NN algorithms which will be introduced in the Deep Learning class
