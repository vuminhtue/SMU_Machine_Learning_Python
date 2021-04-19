---
title: "Regularization and Variable Selection"
teaching: 20
exercises: 0
questions:
- "Why do we need Regularization and Variable Selection in ML model"
objectives:
- "Learn how to apply Regularization and Variable selection in ML model"
keypoints:
- "Regularization, Ridge Regression, LASSO, Elastic Nets"
---
![image](https://user-images.githubusercontent.com/43855029/114340188-ff57bc80-9b24-11eb-826a-69cb444687d4.png)
- One of the major aspects of training your machine learning model is to avoid overfitting (Using more parameter to best fit the training but on the other hand, failed to evaluate the testing).
- The concept of balancing bias and variance, is helpful in understanding the phenomenon of overfitting

## Regularization
- In order to reduce the Model Complexity or to avoid Multi-Collinearity, one needs to reduce the number of covariates 
(or set the coefficient to be zero).
- If the coefficients are too large, let’s penalize them to enforce them to be smaller
- Regularization is a form of multilinear regression, that constrains/regularizes or shrinks the coefficient estimates towards zero.
- In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting
- A simple Multi-Linear Regression look like this:
![image](https://user-images.githubusercontent.com/43855029/114416230-766d6f00-9b7e-11eb-800b-2b7a65782859.png)

=> in which: **β** represents the coefficient estimates for different variables or predictors(x)

The residual sum of squares **RSS** is the loss function of the fitting procedure.
And we need to determine the optimal coefficients **𝛽** to minimize the loss function

![image](https://user-images.githubusercontent.com/43855029/114417635-c39e1080-9b7f-11eb-8465-cbb9e0dff39e.png)

This procedure will adjust the **β** based on the training data. 
If there is any noise in training data, the model will not perform well for testing data. Thus, Regularization comes in and regularizes/shrinkage these 𝛽 towards zero.

There are 3 main types of Regularization. 
- Ridge Regression
- LASSO
- Elastics Nets

### Ridge Regression
![image](https://user-images.githubusercontent.com/43855029/114440609-58ad0380-9b98-11eb-8dd5-643428f60c31.png)

**𝜆**: Regularization Penalty, to be selected that the model minimized the error

The Ridge Regression loss function contains 2 elements: (1) RSS is actually the Ordinary Least Square (OLS) function for MLR and (2) The regularization term with **𝜆**:

![image](https://user-images.githubusercontent.com/43855029/114422155-04982400-9b84-11eb-9f87-65a3d7aec3f3.png)
- Selecting good **𝜆** is essential. In this case, Cross Validation method should be used
- Ridge Regression enforces **β** to be lower but not 0. By doing so, it will not get rid of irrelevant features but rather minimize their impact on the trained model.
- In statistics the coefficient esimated produced by this method is know as **L2 norm**
- It is good practice to normalize predictors to the same sacle before performing Ridge Regression (Because in OLS, the coefficients are scale equivalent)

#### Implementation
Setting up training/testing model:
```python
import pandas as pd
data=pd.read_csv("https://raw.githubusercontent.com/vuminhtue/Machine-Learning-Python/master/data/prostate_data.csv")
ind_train = data["train"]=="T"
data = data.drop(["train"],axis=1)
X_train = data.drop(["lpsa"],axis=1)[ind_train]
X_test = data.drop(["lpsa"],axis=1)[~ind_train]
y_train = data["lpsa"][ind_train]
y_test = data["lpsa"][~ind_train]
```
Predict using Ridge Regression method:
```python
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
n_lambda = 200
lambdas = np.logspace(-2,6, n_lambda)
coefs = []
for ld in lambdas:
    model_RR = Ridge(alpha=ld, fit_intercept=False)
    model_RR.fit(X, y)
    coefs.append(model_RR.coef_)


ax = plt.gca()

ax.plot(lambdas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

```
![image](https://user-images.githubusercontent.com/43855029/114437356-70828880-9b94-11eb-9463-ca9d33b20746.png)

- The plot shows the Mean Square Error based on training model with **𝜆** variation. 
- Top of the chart shows number of predictors used.
- There are 2 **𝜆** values: (1) **𝜆.min** which can be computed using `log(cvfit_Ridge$lambda.min)` and (2) **𝜆.1se** (1 standard error from min value) which can be computed using `log(cvfit_Ridge$lambda.1se)`
- The **β** values for each predictors can be found using `coef(cvfit_Ridge,s=cvfit_Ridge$lambda.1se) or coef(cvfit_Ridge,s=cvfit_Ridge$lambda.min)`

```r
Fit_Ridge <- glmnet(x,y,alpha=0,standardize = TRUE)
plot_glmnet(Fit_Ridge,label=TRUE,xvar="lambda",
            col=seq(1,8),grid.col = 'lightgray')

xtest <- testing[,-c(9,10)]
xtest <- as.matrix(xtest)
```

![image](https://user-images.githubusercontent.com/43855029/114437734-ef77c100-9b94-11eb-94ac-df2794777c81.png)

The plot shows different coefficients for all predictors with **𝜆** variation.
Using **𝜆.1se**, we obtain reasonable result.

```r
> predict_Ridge <- predict(cvfit_Ridge,newx=xtest,s="lambda.1se")
> cor.test(predict_Ridge,testing$lpsa)

	Pearson's product-moment correlation

data:  predict_Ridge and testing$lpsa
t = 5.5527, df = 28, p-value = 6.138e-06
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.4919694 0.8599222
sample estimates:
      cor 
0.7239285 

> postResample(predict_Ridge,testing$lpsa)
     RMSE  Rsquared       MAE 
0.7365551 0.5240725 0.5499648
```
- Ridge Regression's pros: the pros of RR method over OLS is rooted in the bias variance trade-off. As when **𝜆** increases, the flexibility of RR fit decreases, hence decrease the variance but increase the bias
- Ridge Regression's cons: **β** never be 0, so all predictors are included in the final model. Therefore, it is not good for best feature selection.

### LASSO: Least Absolute Shrinkage & Selection Operator
![image](https://user-images.githubusercontent.com/43855029/114440016-a4ab7880-9b97-11eb-8a57-b112cd78f785.png)

- In order to overcome the cons issue in Ridge Regression, the LASSO is introduced with the similar shrinkage parameter, but the different is not in square term of the coefficient but only absolute value
- Similar to Ridge Regression, LASSO also shrink the coefficient, but **force** coefficients to be equal to 0. Making it ability to perform **feature selection**
- In statistics the coefficient esimated produced by this method is know as **L1 norm**

#### Implementation 
```r
cvfit_LASSO    <- cv.glmnet(x,y,alpha=1)
plot(cvfit_LASSO)

log(cvfit_LASSO$lambda.min)
log(cvfit_LASSO$lambda.1se)

coef(cvfit_LASSO,s=cvfit_LASSO$lambda.min)
coef(cvfit_LASSO,s=cvfit_LASSO$lambda.1se)
```
![image](https://user-images.githubusercontent.com/43855029/114452867-eb549f00-9ba6-11eb-9cb4-fddb2a3d69c2.png)

- The plot shows the Mean Square Error based on training model with **𝜆** variation. 
- Top of the chart shows number of predictors used. Now instead of showing all **8** predictors as in Ridge Regression, LASSO shows the different number of predictors as MSE values change. 
- Similar to RR, there are 2 **𝜆** values: (1) **𝜆.min** which can be computed using `log(cvfit_LASSO$lambda.min)` and (2) **𝜆.1se** (1 standard error from min value) which can be computed using `log(cvfit_LASSO$lambda.1se)`
- The corresponding **β** values for each predictors can be found using `coef(cvfit_Ridge,s=cvfit_LASSO$lambda.1se) or coef(cvfit_LASSO,s=cvfit_Ridge$lambda.min) `

```r
Fit_LASSO <- glmnet(x,y,alpha=1)
plot_glmnet(Fit_LASSO,label=TRUE,xvar="lambda",
            col=seq(1,8),,grid.col = 'lightgray')
```
![image](https://user-images.githubusercontent.com/43855029/114453819-e80de300-9ba7-11eb-876c-fba761a277ef.png)
The plot shows different coefficients for all predictors with **𝜆** variation. Depending on **𝜆** values that the **β** varying and it can be 0 at certain point.

Using **𝜆.1se**, we obtain reasonable result:
```r
> predict_LASSO <- predict(cvfit_LASSO,newx=xtest,s="lambda.1se")
> postResample(predict_LASSO,testing$lpsa)
     RMSE  Rsquared       MAE 
0.6783357 0.6096333 0.5030956 
```
### Elastic Nets
Elastic Nets Regularization is a method that includes both LASSO and Ridge Regression. Its formulation for the loss function is as following:
![image](https://user-images.githubusercontent.com/43855029/114456877-615b0500-9bab-11eb-9298-028fcffc03ab.png)

- 𝛼=0: pure Ridge Regression
- 𝛼=1: pure LASSO
- 0<𝛼<1: Elastic Nets

#### Implementation 
```r
cvfit_ELN    <- cv.glmnet(x,y,alpha=0.5)
plot(cvfit_ELN)
```


