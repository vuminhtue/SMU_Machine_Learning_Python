---
title: "Kaggle online competition: Supervised Learning"
teaching: 20
exercises: 0
questions:
- "How to participate in a Kaggle online compeition"
objectives:
- "Download Kaggle data and apply some algorithm technique that you have learnt to solve the actual data"
keypoints:
- "Kaggle"
---
# 10. Kaggle online competition: Supervised Learning
 
This is a perfect competition for data science students who have completed an online course in machine learning and are looking to expand their skill set before trying a featured competition. 

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
 
![image](https://user-images.githubusercontent.com/43855029/156053760-007e3d08-3472-47e5-ba96-c07d8d3fa325.png)

_**Project description:**_

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home. 


For simpilicity: I downloaded the data for you and put it here:
https://github.com/vuminhtue/SMU_Data_Science_workflow_R/tree/master/data/Kaggle_house_prices


## 10.1 Understand the data

There are 4 files in this folder: 
- train.csv: the trained data with 1460 rows and 81 columns. The last column "**SalePrice**" is for output with continuous value
- test.csv: the test data with 1459 rows and 80 columns. Note: There is no  "**SalePrice**" in the last column
- data_description.txt: contains informations on all columns
- sample_submission.csv: is where you save the output from model prediction and upload it to Kaggle for competition

**Objective:**
- We will use the **train.csv**__ data to create the actual train/test set and apply several algorithm to find the optimal ML algorithm to work with this data
- Once model built and trained, apply to the **test.csv**__ and create the output as in format of sample_submission.csv
- Write all analyses in Rmd format.

## 10.2 Create the content with following Data Science workflow:

### Step 1: Load library, Load data

```python
import pandas as pd
import numpy as np
df_train = pd.read_csv("https://raw.githubusercontent.com/vuminhtue/SMU_Data_Science_workflow_R/master/data/Kaggle_house_prices/train.csv")
df_test = pd.read_csv("https://raw.githubusercontent.com/vuminhtue/SMU_Data_Science_workflow_R/master/data/Kaggle_house_prices/test.csv") 

df_train.head()
```

### Step 2: Select variables.

First split the input variables into numerical and categorical (string) values:

<details><summary>Solution</summary>
```python
df_numerical=df_train.select_dtypes(exclude=['object'])
df_categorical=df_train.select_dtypes(include=['object'])
```
</details> 


Visualize  the cross correlation for all numerical input and output **SalePrice**:

Here, we plot the heatmap and retain only the cross corelation higher than 0.6

<details><summary>Solution</summary>

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 10))
sns.heatmap(numerical.corr(), cmap='RdYlGn_r', annot=True,mask = (np.abs(numerical.corr()) < 0.6))
```
</details>
 
- What do you see from the heatmap?
- For now, given only the input values as numerical, what input values would you choose to predict the output **SalePrice**?

<details><summary>Solution</summary>

```python
df_train1 = df_train[["OverallQual","TotalBsmtSF","1stFlrSF","GrLivArea","GarageCars","GarageArea","SalePrice"]]
```

 </details> 
 
### Step 3: Create partition for the data

<details><summary>Solution</summary>

```python
X = df_train1.iloc[:,0:6]
y = df_train1.iloc[:,-1] 
```
</details>
 
### Step 4: Apply 1 ML algorithm to the data and calculate prediction

<details><summary>Solution using Random Forest</summary>
```python
from sklearn.ensemble import RandomForestRegressor
model_RF = RandomForestRegressor(n_estimators=100).fit(X_train,y_train)
y_pred_RF = model_RF.predict(X_test)

```
</details>
 
### Step 5: Evaluate the model output

 <details><summary>Solution using Random Forest</summary>
```python
from sklearn import metrics
print("R2 using Random Forest is: %1.2f " % metrics.r2_score(y_test,y_pred_RF)) 
print("RMSE using Random Forest is: %1.2f" % metrics.mean_squared_error(y_test,y_pred_RF,squared=False))
```
</details>
