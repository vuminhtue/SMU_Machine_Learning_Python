---
title: "05-Supervised Learning with continuous output"
teaching: 20
exercises: 0
questions:
- "How to train a Machine Learning model with continuos output"
objectives:
- "Learn to use different ML algorithm for Supervised Learning"
keypoints:
- "Supervised Learning with continuous output"
---
# 5 Supervised Learning with continuous output

For this session, we gonna use several Machine Learning algorithm  to work with continuous output the supervised learning problem.
First of all, let's import the data:

## 5.1 Preprocessing

### 5.1.1 Import data

Let use the **california housing** data in previous episodes:

```python
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

# Predictors/Input:
X = pd.DataFrame(data.data,columns=data.feature_names)

# Predictand/output:
y = pd.DataFrame(data.target,columns=data.target_names)

print(X.head())
print(y.head())
```

### 5.1.2 Check missing data

```python
print(X.isnull().sum())
print(y.isnull().sum())
```

Since there is no missing data, we move on to the next step:

### 5.1.3 Split model into training & testing set with 60% for training:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.6,random_state=123)
```

Now the input data is ready for supervised learning model, let's select several ML algorithms to work with:

## 5.2 Machine Learning algorithm with Linear Regression

### 5.2.1 Train model using Linear Regression with 1 predictor (for example Medium Income)

#### Fit a Linear model using LinearRegression model:

```python
from sklearn.linear_model import LinearRegression
model_linreg1 = LinearRegression().fit(pd.DataFrame(X_train['MedInc']),y_train)
```

##### Apply trained model to testing data set and evaluate output using R-squared:

```python
from sklearn import metrics
y_pred = model_linreg1.predict(pd.DataFrame(X_test['MedInc']))
print("R2 is: %1.2f" % metrics.r2_score(y_test,y_pred)) 
print("RMSE is: %1.2f" % metrics.mean_squared_error(y_test,y_pred,squared=False)) 
```

the result is:

```
R2 is: 0.48
RMSE is: 0.84
```

We see that using 1 predictor/input, we obtain the output with corresponding R2 of 0.48 and RMSE = 0.84, which is not good enough. (The good R2 should be more than 0.7)
Therefore, we change the approach, still using Linear Regression but with more inputs:

### 5.2.2 Train model using Multi-Linear Regression (with 2 or more predictors)
In this section, we will build the model with 4 inputs ["MedInc","HouseAge","AveRooms","Population"]

#### Fit the training set and predict using test set

```python
model_linreg = LinearRegression().fit(X_train[["MedInc","HouseAge","AveRooms","Population"]],y_train)
y_pred2 = model_linreg.predict(X_test[["MedInc","HouseAge","AveRooms","Population"]])

print("R2 is: %1.2f" % metrics.r2_score(y_test,y_pred2))
print("RMSE is: %1.2f"  % metrics.mean_squared_error(y_test,y_pred2,squared=False)) 
```

Output is therefore better with smaller RMSE and higher Rsquared:

```
R2 is: 0.52
RMSE is: 0.80
```

Still the model outcome is not good enough, so we try another algorithm:

### 5.2.3 Train model using Polynomial Regression

We can slightly improve this by using Polynomial Regression
![image](https://user-images.githubusercontent.com/43855029/115059030-f7e13c00-9eb3-11eb-9887-52461d7a87aa.png)

#### Preprocessing: polynomial regression with `degree of freedom=2`
the degree-2 polynomial features for 2 inputs (a & b) are [1, a, b, a^2, ab, b^2].

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X[["MedInc","HouseAge","AveRooms","Population"]])

X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X_poly),y, train_size=0.6,random_state=123)

print(X_poly.shape)
```

We see that using 4 inputs data, with Polynomial regression, we have 15 input variables [1	a	b	c	d	a2	ab	ac	ad	b2	bc	bd	c2	cd	d2]

#### Fit the new dataset and predict output:

```python
model_linreg_poly = LinearRegression().fit(X_train,y_train)
y_pred_poly = model_linreg_poly.predict(X_test)

print("R2 is: %1.2f " % metrics.r2_score(y_test,y_pred_poly)) 
print("RMSE is: %1.2f" % metrics.mean_squared_error(y_test,y_pred_poly,squared=False))
```

The output is even better with R2 for testing data is 0.55 and lower RMSE.

```
R2 is: 0.55 
RMSE is: 0.78
```

The **R2=0.55** shows improvement using polynomial regression!

How about using more degrees of freedom?


#### Polynomial regression with `degree of freedom=4`

Can we improve the result with more degree of freedome? Let's try using df=4:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X[["MedInc","HouseAge","AveRooms","Population"]])

X_train4, X_test4, y_train4, y_test4 = train_test_split(pd.DataFrame(X_poly),y, train_size=0.6,random_state=123)

model_linreg_poly4 = LinearRegression().fit(X_train4,y_train4)
y_pred_poly4 = model_linreg_poly4.predict(X_test4)

print("R2 for 4 dof is: %1.2f " % metrics.r2_score(y_test4,y_pred_poly4)) 
print("RMSE for 4 dof is: %1.2f" % metrics.mean_squared_error(y_test4,y_pred_poly4,squared=False)) 

```

the output is:

```
R2 for 4 dof of testing is: -4.33 
RMSE for 4 dof of testing is: 2.68
```

The R2 in sklearn can be negative, it arbitrarily means that the model is worse. More info on sklearn [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

Why increasing the degree of freedom, my results getting worst?
It's called OVERFITTING

## 5.3 Overfitting

Overfitting occurs when we used lots of unesscessary input data for training process. It fits the training data so well that it is worse when applied to testing data:

![image](https://user-images.githubusercontent.com/43855029/153645935-b9eeebe5-424a-490a-aa95-006088a66b21.png)

Exercise 1: Let use all dataset to train the data to see if using all input data, we have overfitting?

```python

```

Exercise 2: Let's check the R2 and RMSE for training set using 2 and 4 degree of freedom to see if the 4 dof is better than 2 dof in fitting back to training data?

```python

```

## 5.4 Other Supervised ML algorithm for continuous data

There are many other ML algorithm that helps to overcome the issue of overfitting, for example:

### 5.4.1 Decision Tree

- Tree based learning algorithms are considered to be one of the best and mostly used supervised learning methods.
- Tree based methods empower predictive models with high accuracy, stability and ease of interpretation
- Non-parametric and non-linear relationships
- Types: Continuous (DecisionTreeRegressor) and Categorical (DecisionTreeClassifier)

![image](https://user-images.githubusercontent.com/43855029/153648313-da3a9a08-c4ad-48c9-bebd-df34f1651f98.png)

Let use all data in this exercise, the Decision Tree algorithm for continuous output in sklearn is called **DecisionTreeRegressor**

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.6,random_state=123)

from sklearn.tree import DecisionTreeRegressor
model_DT = DecisionTreeRegressor(max_depth=6).fit(X_train,y_train)
y_pred_DT = model_DT.predict(X_test)

print("R2 using Decision Tree is: %1.2f " % metrics.r2_score(y_test,y_pred_DT)) 
print("RMSE using Decision Tree is: %1.2f" % metrics.mean_squared_error(y_test,y_pred_DT,squared=False))
```

output:

```
R2 using Decision Tree is: 0.65 
RMSE using Decision Tree is: 0.68
```

Now we can see that Decision Tree helps to overcome the overfitting by trimming down the unnecessary input data.

#### Visualization the Decision Tree:

The following required graphviz model to be loaded when you requested for a Python Notebook.

![image](https://user-images.githubusercontent.com/43855029/153649826-000cc8ab-dfb9-43b7-b31c-26ecaf03d0a1.png)

```python
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(model_DT, out_file=None,                      
                      filled=True, rounded=True,
                      feature_names=data.feature_names,
                      special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
```

### 5.4.2 Random Forest

![image](https://user-images.githubusercontent.com/43855029/153650870-13494bba-d440-4006-b98a-6fb1509d10d5.png)


- Random Forest is considered to be a panacea of all data science problems. On a funny note, when you can’t think of any algorithm (irrespective of situation), use random forest!
- Opposite to Decision Tree, Random Forest use bootstrapping technique to grow multiple tree
- Random Forest is a versatile machine learning method capable of performing both regression and classification tasks.
- It is a type of ensemble learning method, where a group of weak models combine to form a powerful model.
- The end output of the model is like a black box and hence should be used judiciously.


![image](https://user-images.githubusercontent.com/43855029/153650921-ecc70313-6e17-4bb6-92cb-bab11a39ab0c.png)

```python
from sklearn.ensemble import RandomForestRegressor
model_RF = RandomForestRegressor(n_estimators=10).fit(X_train,y_train)
y_pred_RF = model_RF.predict(X_test)

print("R2 using Random Forest is: %1.2f " % metrics.r2_score(y_test,y_pred_RF)) 
print("RMSE using Random Forest is: %1.2f" % metrics.mean_squared_error(y_test,y_pred_RF,squared=False))
```

Here we use n=10 estimators (growing using n trees in the forest) and The output is much better:

```
R2 using Random Forest is: 0.81 
RMSE using Random Forest is: 0.51
```

## 5.5 Ensemble Machine Learning

- Ensemble is a method in Machine Learning that combine decision from several ML models to obtain optimum output.
- Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. Unlike a statistical ensemble in statistical mechanics, which is usually infinite, a machine learning ensemble consists of only a concrete finite set of alternative models, but typically allows for much more flexible structure to exist among those alternatives
- The bonus point when applying both Bagging and Boosting in sklearn that they can be run in parallel!

**Types of Ensembles:**

There are 2 main types of Ensembles in ML:

- Bagging: Boostrap Aggregation

![image](https://user-images.githubusercontent.com/43855029/153652070-c067fc10-6322-49d1-92ed-b27532af11b6.png)

Random Forest is considered Bagging Ensemble method!

- Boosting: Boost the weak predictors

![image](https://user-images.githubusercontent.com/43855029/153652096-4e93d213-58b9-4b27-88fa-e8b42a9cd6e5.png)


### 5.5.1 Bagging with RandomForest

We can apply Bagging to different ML algorithm like Linear Regression, Decision Tree, Random Forest, etc. 
Following are the syntax:

```python
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

model_RF = RandomForestRegressor()

model_bag_RF = BaggingRegressor(base_estimator=model_RF, n_estimators=100,
                            bootstrap=True, n_jobs=-1,
                            random_state=123)
                            
model_bag_RF.fit(X_train, y_train)

y_pred_bagRF = model_bag_RF.predict(X_test)

print("R2 using Bagging Random Forest is: %1.2f " % metrics.r2_score(y_test,y_pred_bagRF)) 
print("RMSE using Baggin Random Forest is: %1.2f" % metrics.mean_squared_error(y_test,y_pred_bagRF,squared=False))
```

Note that here we use n_estimators = 100 for bagging model (it grows 100 times the RandomForest model).
The n_jobs=-1 means that it utilizes all the cores inside a compute nodes that we have

And the output is very similar to RandomForest:

```
R2 using Bagging Random Forest is: 0.80 
RMSE using Baggin Random Forest is: 0.51
```

Let's try with some Boosting Ensemble approach:

### 5.5.2 Boosting with Adaboost

```python
from sklearn.ensemble import AdaBoostRegressor
model_ADA = AdaBoostRegressor(n_estimators=100, learning_rate=0.03).fit(X_train, y_train)
y_pred_ADA = model_ADA.predict(X_test)

print("R2 using Adaboost is: %1.2f " % metrics.r2_score(y_test,y_pred_ADA)) 
print("RMSE using Adaboost is: %1.2f" % metrics.mean_squared_error(y_test,y_pred_ADA,squared=False))
```

The output is not as good as Bagging RF

```
R2 using Adaboost is: 0.59 
RMSE using Adaboost is: 0.75
```

### 5.5.3 Gradient Boosting Machine

```python
from sklearn.ensemble import GradientBoostingRegressor
model_GBM = GradientBoostingRegressor(n_estimators=100).fit(X_train,y_train)
y_pred_GBM = model_GBM.predict(X_test)

print("R2 using GBM is: %1.2f " % metrics.r2_score(y_test,y_pred_GBM)) 
print("RMSE using GBM is: %1.2f" % metrics.mean_squared_error(y_test,y_pred_GBM,squared=False))
```

The output is better than Adaboost:

```
R2 using GBM is: 0.79 
RMSE using GBM is: 0.53
```

### Which is better in Ensemble? Bagging or Boosting?

![image](https://user-images.githubusercontent.com/43855029/153654625-d7efe94d-1fc4-4ee6-9b4b-f897f52a909e.png)

Ensemble overcome the limitation of using only single model
Between bagging and boosting, there is no better approach without trial & error.

## 5.2 For categorical output
### 5.2.1 Train model using Logistic Regression
- Logistic regression is another technique borrowed by machine learning from the field of statistics. It is the go-to method for binary classification problems (problems with two class values).
- Typical binary classification: True/False, Yes/No, Pass/Fail, Spam/No Spam, Male/Female
- Unlike linear regression, the prediction for the output is transformed using a non-linear function called the logistic function.
- The standard logistic function has formulation:

![image](https://user-images.githubusercontent.com/43855029/114233181-f7dcbb80-994a-11eb-9c89-58d7802d6b49.png)

![image](https://user-images.githubusercontent.com/43855029/114233189-fb704280-994a-11eb-9019-8355f5337b37.png)

In this example, we create a sample data set and use logistic regression to solve it. The example is taken from [here](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

Load library and create sample data set:

```python
from sklearn.datasets import make_classification

# generate sample data
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
```

Partitioning Data to train/test:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
```

Train model using Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model_LogReg = LogisticRegression().fit(X_train, y_train)
y_pred = model_LogReg.predict(X_test)

from sklearn.linear_model import LogisticRegression
model_LogReg = LogisticRegression().fit(X_train, y_train)
# predict output:
y_pred = model_LogReg.predict(X_test)
# predict probabilities
lr_probs = model_LogReg.predict_proba(X_test)
```

Evaluate output with accurary level:
```python
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)
```
We retrieve the **accuracy = 0.834**

Now compute AUC-ROC and plot curve

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# generate a no skill prediction (majority class)
ns_probs = np.zeros(len(y_test))

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs[:,1])
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs[:,1])
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/120822169-22e72400-c524-11eb-97fe-46f711a11072.png)

An alternative way to plot AUC-ROC curve, using additional toolbox ["scikit-plot"](https://scikit-plot.readthedocs.io/en/stable/)
```python
pip install scikit-plot
```

The shorter code for using this library:

```python
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, lr_probs)
plt.show()
```

![image](https://user-images.githubusercontent.com/43855029/120822378-588c0d00-c524-11eb-9cdc-431bd927ad48.png)

