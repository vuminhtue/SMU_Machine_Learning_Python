---
title: "Training Machine Learning model using Regression Method"
teaching: 20
exercises: 0
questions:
- "How to train a Machine Learning model using Regression method"
objectives:
- "Learn to use different Regression algorithm for Machine Learning training"
keypoints:
- "Regression training"
---
# Supervised Learning training
## Train model using Linear Regression with 1 predictor
Let use the **airquality** data in previous episodes:

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics

data_df = pd.DataFrame(pd.read_csv('https://raw.githubusercontent.com/vuminhtue/Machine-Learning-Python/master/data/r_airquality.csv'))

imputer = KNNImputer(n_neighbors=2, weights="uniform")
data_knnimpute = pd.DataFrame(imputer.fit_transform(data_df))
data_knnimpute.columns = data_df.columns

X_train, X_test, y_train, y_test = train_test_split(data_knnimpute['Temp'],
                                                    data_knnimpute['Ozone'],
                                                    train_size=0.6,random_state=123)
```
Fit a Linear model using `method=lm`
```python
from sklearn.linear_model import LinearRegression
model_linreg = LinearRegression().fit(X_train[:,None],y_train)
```
Apply trained model to testing data set and evaluate output using R-squared:
```python
y_pred = model_linreg.predict(X_test[:,None])
metrics.r2_score(y_test,y_pred) # R^2
metrics.mean_squared_error(y_test,y_pred,squared=False) #RMSE
```

## Train model using Multi-Linear Regression (with 2 or more predictors)
From the above model, the **R2=0.39**:

The reason is that we only build the model with 1 input `Temp`.
In this section, we will build the model with more input `Solar Radiation, Wind, Temperature`:
```r
X_train, X_test, y_train, y_test = train_test_split(data_knnimpute[['Temp','Wind','Solar.R']],
                                                    data_knnimpute['Ozone'],
                                                    train_size=0.6,random_state=123)
model_linreg = LinearRegression().fit(X_train,y_train)
y_pred2 = model_linreg.predict(X_test)

metrics.r2_score(y_test,y_pred)
metrics.mean_squared_error(y_test,y_pred,squared=False)
```
Output is therefore better with smaller RMSE and higher Rsquared at **0.5**

## Train model using Polynomial Regression
From Multi-Linear Regression, the best **R2=0.5** using 3 predictors.
We can slightly improve this by using Polynomial Regression
![image](https://user-images.githubusercontent.com/43855029/115059030-f7e13c00-9eb3-11eb-9887-52461d7a87aa.png)

In this study, let use polynomial regression with `degree of freedom=2`
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(data_knnimpute[['Temp','Wind','Solar.R']])
X_train, X_test, y_train, y_test = train_test_split(X_train_poly,
                                                    data_knnimpute['Ozone'],
                                                    train_size=0.6,random_state=123)
model_linreg_poly = LinearRegression().fit(X_train,y_train)
y_pred_poly = model_linreg_poly.predict(X_test)
metrics.r2_score(y_test,y_pred_poly)
```
The **R2=0.58** shows improvement using polynomial regression!

## Train model using Logistic Regression
- Logistic regression is another technique borrowed by machine learning from the field of statistics. It is the go-to method for binary classification problems (problems with two class values).
- Typical binary classification: True/False, Yes/No, Pass/Fail, Spam/No Spam, Male/Female
- Unlike linear regression, the prediction for the output is transformed using a non-linear function called the logistic function.
- The standard logistic function has formulation:

![image](https://user-images.githubusercontent.com/43855029/114233181-f7dcbb80-994a-11eb-9c89-58d7802d6b49.png)

![image](https://user-images.githubusercontent.com/43855029/114233189-fb704280-994a-11eb-9019-8355f5337b37.png)

In this example, we use `breast cancer` data set built-in [sklearn data](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer).

This is a data set that classify breast cancer to `malignant` or `benign` based on different input data on the breast's measurement from 569 patients

First import necessary package:
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.Linear_Model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
```
Read in data:
```python
datab = load_breast_cancer()
datab_df = pd.DataFrame(data=datab.data,columns=datab.feature_names)
features = datab['feature_names']
datab_df['target'] = datab.target
datab_df["target_name"] = datab_df['target'].map({i:name for i,name in enumerate(datab.target_names)})
datab_df.sample(5)
```
Standardize input data:
```python
data_std = pd.DataFrame(scale(datab_df.loc[:,features],axis=0, with_mean=True, with_std=True, copy=True))
```
Partitioning Data to train/test:
```python
X_train, X_test, y_train, y_test = train_test_split(data_std,
                                                    datab_df['target_name'],
                                                    train_size=0.6,random_state=123)
```
Train model using Logistic Regression
```python
model_LogReg = LogisticRegression().fit(X_train, y_train)
y_pred = model_LogReg.predict(X_test)
```
Evaluate output with accurary level:
```python
metrics.accuracy_score(y_test,y_pred)
```
We retrieve the **accuracy = 0.99**
