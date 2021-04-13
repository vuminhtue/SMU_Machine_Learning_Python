---
title: "Introduction to Scikit Learn"
teaching: 40
exercises: 0
questions:
- "What is Scikit Learn"
objectives:
- "Master Scikit Learn for Machine Learning"
keypoints:
- "sklearn"
---

## What is Scikit-Learn
![image](https://user-images.githubusercontent.com/43855029/114609814-30db9f80-9c6d-11eb-8d4e-781f578e1d79.png)

- Scikit-learn is probably the most useful library for machine learning in Python. 
- The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
- The sklearn package contains tools for:
```
- data splitting
- pre-processing
- feature selection
- model tuning using resampling
- variable importance estimation
as well as other functionality.
```
## Install `sklearn`
- In local Anaconda Navigator:
`Select Environment => base (root) => In *not installed* search for **scikit-learn** package to install (apply).`

- In Linux conda management:
`Create conda package management then run
$ pip3 install -U scikit-learn`


## Pre-processing using `caret`
There are several steps that we will use `caret` for. For preprocessing raw data, we gonna use `caret` in these tasks:
- Visualize important variables
- Preprocessing with missing value
- Preprocessing: transform data
- Data partition: training and testing

### Visualize important variables
Here we introduce the library `GGally`  with function `ggpairs` to help user in visualizing the input data
```r
library(GGally)
ggpairs(data=iris,aes(colour=Species))
```
![image](https://user-images.githubusercontent.com/43855029/114196055-01e8c500-991f-11eb-8eaf-816f25e6c534.png)

### Pre-processing with missing value
- Most of the time the input data has missing values (`NA, NaN, Inf`) due to data collection issue (power, sensor, personel). 
- There are three main problems that missing data causes: missing data can introduce a substantial amount of bias, make the handling and analysis of the data more arduous, and create reductions in efficiency
- These missing values need to be treated/cleaned before we can use because "Garbage in => Garbage out".
- There are several ways to treat the missing values:
- Method 1: remove all missing `NA` values
```r
library(base)
data("airquality") # Here we use this sample data because it contains missing value
new_airquality1 <- na.omit(airquality)
``` 
- Method 2: Set `NA` to mean value 
```r
NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))
new_airquality2 <-replace(airquality, TRUE, lapply(airquality, NA2mean))
```
- Method 3: Use `Impute` to handle missing values
In statistics, imputation is the process of replacing missing data with substituted values. Because missing data can create problems for analyzing data, imputation is seen as a way to avoid pitfalls involved with listwise deletion of cases that have missing values. That is to say, when one or more values are missing for a case, most statistical packages default to discarding any case that has a missing value, which may introduce bias or affect the representativeness of the results. Imputation preserves all cases by replacing missing data with an estimated value based on other available information. Once all missing values have been imputed, the data set can then be analysed using standard techniques for complete data. There have been many theories embraced by scientists to account for missing data but the majority of them introduce bias. A few of the well known attempts to deal with missing data include: hot deck and cold deck imputation; listwise and pairwise deletion; mean imputation; non-negative matrix factorization; regression imputation; last observation carried forward; stochastic imputation; and multiple imputation.

Here we use `preProcess` function from `caret` to perform `bagImpute` (Bootstrap Aggregation Imputation):
```r
library(caret)
PreImputeBag <- preProcess(airquality,method="bagImpute")
DataImputeBag <- predict(PreImputeBag,airquality)
```
In addition to `bagImpute`, we also can use `knnImpute` (K-Nearest Neighbour Imputation)
`knnImpute` can also be used to normalized the data
```r
library(gridExtra)
PreImpute <- preProcess(airquality[,-c(5,6)],method="knnImpute")
TraImpute <- predict(PreImpute,airquality[,-c(5,6)])

plot1 <- ggplot(airquality,aes(Ozone)) + geom_histogram(bins=30)+labs(title="Original Probability")
plot2 <- ggplot(TraImpute,aes(Ozone)) + geom_histogram(bins=30)+labs(title="KnnImpute Transform to Normal Distribution")
grid.arrange(plot1,plot2,nrow=2)
```
![image](https://user-images.githubusercontent.com/43855029/114202025-cd780780-9924-11eb-999f-b89c9080cfdf.png)

**Note** 
`bagImpute` is more powerful and computational cost than `knnImpute`

### Pre-processing with Transforming data
#### Using Standardization
![image](https://user-images.githubusercontent.com/43855029/114231774-df6ba180-9948-11eb-9c61-3d2e0d3df889.png)

- Standardization comes into picture when features of input data set have large differences between their ranges, or simply when they are measured in different measurement units for example: rainfall (0-1000mm), temperature (-10 to 40oC), humidity (0-100%), etc.
- Standardition Convert all independent variables into the same scale (mean=0, std=1) 
- These differences in the ranges of initial features causes trouble to many machine learning models. For example, for the models that are based on distance computation, if one of the features has a broad range of values, the distance will be governed by this particular feature.
- The example below use data from above:
```r
PreStd <- preProcess(training[,-c(1,5,6)],method=c("center","scale")) 
TrainStd <- predict(PreStd,training[,-c(1,5,6)])
apply(TrainStd,2,mean)
apply(TrainStd,2,sd)

TestStd <- predict(PreStd,testing[,-c(1,5,6)])
apply(TestStd,2,mean)
apply(TestStd,2,sd)
```

#### Using Box-Cox Transformation
- A Box Cox transformation is a transformation of a non-normal dependent variables into a normal shape. 
- Normality is an important assumption for many statistical techniques; if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests.
- The Box Cox transformation is named after statisticians George Box and Sir David Roxbee Cox who collaborated on a 1964 paper and developed the technique.
```r
PreBxCx <- preProcess(training[,-c(5,6)],method="BoxCox")
TrainBxCx <- predict(PreBxCx,training[,-c(5,6)])

plot1 <- ggplot(training,aes(Ozone)) + geom_histogram(bins=30)+labs(title="Original Probability")
plot2 <- ggplot(TrainBxCx,aes(Ozone)) + geom_histogram(bins=30)+labs(title="Box-Cox Transform to Normal")
library(gridExtra)
grid.arrange(plot1,plot2,nrow=2)
```
![image](https://user-images.githubusercontent.com/43855029/114201422-298e5c00-9924-11eb-9e40-0b8b45138f46.png)
 
 ### Pre-processing as argument:
 When using Preprocessing as argument in the training process in caret, the method is changed to preProcess, for example:
```r
modelFit2 <- train(Ozone~Temp,data=training,
                  preProcess=c("center","scale","BoxCox"),
                  method="lm")
prediction2 <- predict(modelFit2,testing)
```

## Post-processing - Evaluate the test result
Once getting the prediction coming out from Machine Learning training model, user is ready to evaluate the output with observed data from testing set.
There are 2 main types of output: (1) Continuous data & (2) Discreet/Classification/Categorical data

- For Regression with continuous data
```r
cor(prediction,testing)
cor.test(prediction,testing)
postResample(prediction,testing)
```

- For categorical data
```r
confusionMatrix(predict,testing)
```

We will discuss more in detail in the training section.
