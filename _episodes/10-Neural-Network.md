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

## Neural Network
![image](https://user-images.githubusercontent.com/43855029/114472746-da188c00-9bc0-11eb-913c-9dcd14f872ac.png)
![image](https://user-images.githubusercontent.com/43855029/114472756-dd137c80-9bc0-11eb-863d-7c4d054efa89.png)

- Formulation of Neural Network

![image](https://user-images.githubusercontent.com/43855029/114472776-e997d500-9bc0-11eb-9f70-450389c912df.png)
```
Here, x1,x2....xn are input variables. w1,w2....wn are weights of respective inputs.
b is the bias, which is summed with the weighted inputs to form the net inputs. 
Bias and weights are both adjustable parameters of the neuron.
Parameters are adjusted using some learning rules. 
The output of a neuron can range from -inf to +inf.
The neuron doesn’t know the boundary. So we need a mapping mechanism between the input and output of the neuron. 
This mechanism of mapping inputs to output is known as Activation Function.
```
- Activation functions:
![image](https://user-images.githubusercontent.com/43855029/114575672-6752f380-9c48-11eb-8d53-c78d052cdf17.png)

```python
xrange = np.linspace(-2, 2, 200)

plt.figure(figsize=(7,6))

plt.plot(xrange, np.maximum(xrange, 0), label = 'ReLU')
plt.plot(xrange, np.tanh(xrange), label = 'Hyperbolic Tangent')
plt.plot(xrange, 1 / (1 + np.exp(-xrange)), label = 'Sigmoid')
plt.plot(xrange, xrange, label = 'Linear')
plt.plot(xrange, np.heaviside(xrange, 0.5), label = 'Step')
plt.legend()
plt.title('Neural network activation functions')
plt.xlabel('Input value (x)')
plt.ylabel('Activation function output')

plt.show()
```
![image](https://user-images.githubusercontent.com/43855029/115565946-d3a4a700-a287-11eb-93b8-3209fa182436.png)


- Neural Network formulation: Multi-Layer Perceptron (MLP)
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

### Implementation
Split the data
```python

```

- Fit the MLP Neural Network using **1** hidden layer:
```python
set.seed(123)
ModNN <- neuralnet(mpg~cyl+disp+hp+drat+wt+qsec+carb,trainNN, hidden=10,linear.output = T)
plot(ModNN)
```
![image](https://user-images.githubusercontent.com/43855029/114492632-f0d1d980-9be6-11eb-89c5-196f9f3546d9.png)
```r
#Predict using Neural Network
predictNN <- compute(ModNN,testNN[,c(2:7,11)])
predictmpg<- predictNN$net.result*(smax-smin)[1]+smin[1]
postResample(testing$mpg,predictmpg)
     RMSE  Rsquared       MAE 
3.0444857 0.8543388 2.3276645 
```
