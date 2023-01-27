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

## 2.1 What is Scikit-Learn
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
## 2.2 Install `sklearn`
We have installed kernel **ML_SKLN** which contains the scikit-learn package in M2. More information can be found in the setup page

## 2.3 Pre-processing using `sklearn`
There are several steps that we will use `sklearn` for. For preprocessing raw data, we gonna use `sklearn` in these tasks:
- Preprocessing with missing value
- Preprocessing: transform data

### 2.3.1 Pre-processing with missing value
- Most of the time the input data has missing values (`NA, NaN, Inf`) due to data collection issue (power, sensor, personel). 
- There are three main problems that missing data causes: missing data can introduce a substantial amount of bias, make the handling and analysis of the data more arduous, and create reductions in efficiency
- These missing values need to be treated/cleaned before we can use because "Garbage in => Garbage out".
- There are several ways to treat the missing values:

![image](https://user-images.githubusercontent.com/43855029/153270189-5bf6f452-64ab-4af7-b30d-de985c8c5661.png)
[source](https://www.kaggle.com/parulpandey/a-guide-to-handling-missing-values-in-python)

#### Read in data with missing value and check the missing values:

```python
import pandas as pd
data_df = pd.read_csv('https://raw.githubusercontent.com/vuminhtue/SMU_Machine_Learning_Python/master/data/airquality.csv')
data_df.shape
data_df.head()
data_df.isnull().sum()
``` 
#### Method 1: ignore missing values:

Many function in python ignore the missing values, for example the mean & count function:

```python
data_df['Ozone'].mean() 
data_df['Ozone'].count()
```

You will see that the count function only print 116 values (out of 153 values (including NA) in total) of Ozone columns

#### Method 2: remove entire row with missing `NA` values

```python
data2 = data_df.dropna()
``` 

#### Method 3: drop the entire column (not recommended):

```python
data3 = data_df.drop("Ozone",axis=1)
```

Note: axis = 1 (column), axis = 0 (row)

#### Method 4: Fill `NA` with constant values

Often time, the mising data can be set to 0 or 1 (or any other meaningful data set in your field):
Following code fill the missing value with 0:

```python
data4 = data_df.copy()
data4.fillna(0, inplace=True)
```

#### Method 5: Full `NA` to mean/median/max/min value 
Very similar to filling with constant value:

```python
data5 = data_df.copy()
data5.fillna(data5.mean(), inplace=True)
```

Or using SimpleImputer function from sklearn:

```python
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data5 = pd.DataFrame(imputer.fit_transform(data_df))
data5.columns = data_df.columns
```

**Note:**
SimpleImputer converts missing values to **mean, median, most_frequent and constant**.

#### Method 6: **Advanced** Use KNN-based `Impute` to handle missing values

In statistics, imputation is the process of replacing missing data with substituted values. Because missing data can create problems for analyzing data, imputation is seen as a way to avoid pitfalls involved with listwise deletion of cases that have missing values. That is to say, when one or more values are missing for a case, most statistical packages default to discarding any case that has a missing value, which may introduce bias or affect the representativeness of the results. Imputation preserves all cases by replacing missing data with an estimated value based on other available information. Once all missing values have been imputed, the data set can then be analysed using standard techniques for complete data. There have been many theories embraced by scientists to account for missing data but the majority of them introduce bias. A few of the well known attempts to deal with missing data include: hot deck and cold deck imputation; listwise and pairwise deletion; mean imputation; non-negative matrix factorization; regression imputation; last observation carried forward; stochastic imputation; and multiple imputation.

`knnImpute` can also be used to fill in missing value

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")
data_knnimpute = pd.DataFrame(imputer.fit_transform(data_df))
data_knnimpute.columns = data_df.columns
```

**Note:**
- In addition to KNNImputer, there are **IterativeImputer** (Multivariate imputer that estimates each feature from all the others) and **MissingIndicator**(Binary indicators for missing values)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
Iimputer = IterativeImputer()
data_mice = pd.DataFrame(Iimputer.fit_transform(data_df))
data_mice.columns = data_df.columns
```

- More information on sklearn.impute can be found [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute)

### 2.3.2 Pre-processing with Transforming data
#### 2.3.2.1 Using Standardization
![image](https://user-images.githubusercontent.com/43855029/114231774-df6ba180-9948-11eb-9c61-3d2e0d3df889.png)

- Standardization comes into picture when features of input data set have large differences between their ranges, or simply when they are measured in different measurement units for example: rainfall (0-1000mm), temperature (-10 to 40oC), humidity (0-100%), etc.
- Standardition Convert all independent variables into the same scale (mean=0, std=1) 
- These differences in the ranges of initial features causes trouble to many machine learning models. For example, for the models that are based on distance computation, if one of the features has a broad range of values, the distance will be governed by this particular feature.
- The example below use data from above:

```python
from sklearn.preprocessing import scale
data_std = pd.DataFrame(scale(data_knnimpute,axis=0, with_mean=True, with_std=True, copy=True))
# axis used to compute the means and standard deviations along. If 0, independently standardize each feature, otherwise (if 1) standardize each sample.
data_std.columns = data_knnimpute.columns
data_std
```

#### 2.3.2.2 Using scaling with predefine range
Transform features by scaling each feature to a given range.
This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
Formulation for this is:

```python
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#By default, it scales for (0, 1) range
data_scaler = pd.DataFrame(scaler.fit_transform(data_knnimpute))
data_scaler.columns = data_knnimpute.columns
data_scaler
```

#### 2.3.2.3 Using Box-Cox Transformation

- A [Box Cox](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1964.tb00553.x) transformation is a transformation of a non-normal dependent variables into a normal shape. 
- Normality is an important assumption for many statistical techniques; if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests.
- The Box Cox transformation is named after statisticians George Box and Sir David Roxbee Cox who collaborated on a 1964 paper and developed the technique.
- BoxCox can only be applied to stricly positive values

![image](https://user-images.githubusercontent.com/43855029/191553926-cbdb29bf-cab1-47c7-838d-8243c120106e.png)


```python
from sklearn.preprocessing import power_transform
data_BxCx = pd.DataFrame(power_transform(data_knnimpute.iloc[:,0:4],method="box-cox"))
data_BxCx.columns = data_knnimpute.columns[0:4]
data_BxCx[["Month","Day"]]=data_knnimpute[["Month","Day"]]
data_BxCx
```

#### 2.3.2.4 Using Yeo Johnson Transformation
While BoxCox only works with positive value, a more recent transformation method [Yeo Johnson](https://www.jstor.org/stable/2673623) can transform both positive and negative values
```python
data_yeo_johnson = pd.DataFrame(power_transform(data_knnimpute.iloc[:,0:4],method="yeo-johnson"))
data_yeo_johnson.columns = data_knnimpute.columns[0:4]
data_yeo_johnson[["Month","Day"]]=data_knnimpute[["Month","Day"]]
data_yeo_johnson
```

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

ax1 = plt.subplot(1,2,1)
sns.distplot(data_knnimpute["Ozone"])
ax1.set_title("Original probability")

ax2 = plt.subplot(1,2,2)
sns.distplot(data_BxCx["Ozone"])
ax2.set_title("Box-Cox Transformation")
```

![image](https://user-images.githubusercontent.com/43855029/191554249-8fc8f758-33b9-4ee6-94eb-816052b6f665.png)



