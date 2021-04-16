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

## Train model using Multi-Linear Regression
From the above model, the `R^2` only show the reasonable result 0.39:

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
Output is therefore better with smaller RMSE and higher Rsquared

## Train model using Stepwise Linear Regression
It’s a step by step Regression to determine which covariates set best match with the dependent variable. Using AIC as criteria:

```r
modFit_SLR <- train(Ozone~Solar.R+Wind+Temp,data=training,method="lmStepAIC")
summary(modFit_SLR$finalModel)

prediction_SLR <- predict(modFit_SLR,testing)

cor.test(prediction_SLR,testing$Ozone)
postResample(prediction_SLR,testing$Ozone)
```

```r
> postResample(prediction_SLR,testing$Ozone)
      RMSE   Rsquared        MAE 
25.0004212  0.5239849 17.0977421 
```

## Train model using Principal Component Regression
Linear Regression using the output of a Principal Component Analysis (PCA). 
PCR is skillful when data has lots of highly correlated predictors

```r
modFit_PCR <- train(Ozone~Solar.R+Wind+Temp,data=training,method="pcr")
summary(modFit_PCR$finalModel)

prediction_PCR <- predict(modFit_PCR,testing)

cor.test(prediction_PCR,testing$Ozone)
postResample(prediction_PCR,testing$Ozone)
```

## Train model using Logistic Regression
- Logistic regression is another technique borrowed by machine learning from the field of statistics. It is the go-to method for binary classification problems (problems with two class values).
- Typical binary classification: True/False, Yes/No, Pass/Fail, Spam/No Spam, Male/Female
- Unlike linear regression, the prediction for the output is transformed using a non-linear function called the logistic function.
- The standard logistic function has formulation: ![image](https://user-images.githubusercontent.com/43855029/114233181-f7dcbb80-994a-11eb-9c89-58d7802d6b49.png)

![image](https://user-images.githubusercontent.com/43855029/114233189-fb704280-994a-11eb-9019-8355f5337b37.png)


In this example, we use `spam` data set from package `kernlab`.
This is a data set collected at Hewlett-Packard Labs, that classifies **4601** e-mails as spam or non-spam. In addition to this class label there are **57** variables indicating the frequency of certain words and characters in the e-mail.
More information on this data set can be found [here](https://rdrr.io/cran/kernlab/man/spam.html)

Train the model:
```r
library(kernlab)
data(spam)
names(spam)

indTrain <- createDataPartition(y=spam$type,p=0.6,list = FALSE)
training <- spam[indTrain,]
testing  <- spam[-indTrain,]

ModFit_glm <- train(type~.,data=training,method="glm")
summary(ModFit_glm$finalModel)
```
Predict based on testing data and evaluate model output:
```r
predictions <- predict(ModFit_glm,testing)
confusionMatrix(predictions, testing$type)
```
