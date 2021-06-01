---
title: "Data Partition with Scikit-Learn"
teaching: 20
exercises: 0
questions:
- "What is Data Partition"
objectives:
- "Learn how to split data using sklearn"
keypoints:
- "sklearn, data partition"
---

# Data partition: training and testing

![image](https://user-images.githubusercontent.com/43855029/120378647-b1716080-c2ec-11eb-8693-60defbbad7e2.png)


- In Machine Learning, it is mandatory to have training and testing set. Some time a verification set is also recommended.
Here are some functions for spliting training/testing set in `sklearn`:

- `train_test_split`: create series of test/training partitions
- `Kfold` splits the data into k groups
- `StratifiedKFold` splits the data into k groups based on a grouping factor.
- `RepeatKfold`
- `ShuffleSplit`
- `LeaveOneOut`
- `LeavePOut`

Due to time constraint, we only focus on `train_test_split`, `KFolds` and `StratifiedKFold` 
## Scikit-Learn data
The `sklearn.datasets` package embeds some small toy [datasets](https://scikit-learn.org/stable/datasets.html)
```
For each dataset, there are 4 varibles:
- **data**: numpy array of predictors/X
- **target**: numpy array of predictant/target/y
- **feature_names**: names of all predictors in X
- **target.names**: names of all predictand in y
```
For example:
```python
from sklearn.datasets import load_iris
data = load_iris()
print(data.data)
print(data.target)
print(data.feature_names)
print(data.target_names)
```

In this example we gonna use the renowned iris flower data
```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
```

## Data spliting using `train_test_split`: **Single fold**
Here we use `train_test_split` to randomly split 60% data for training and the rest for testing:
![image](https://user-images.githubusercontent.com/43855029/114209883-22b81700-992d-11eb-83a4-c4ab1538a1e5.png)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.6,random_state=123)
#random_state: int, similar to R set_seed function
```

## Data spliting using `K-fold`
- The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. 
- As such, the procedure is often called k-fold cross-validation. 
- When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation.
- Kfold method returns the order of the samples chosen for train and test sets in each fold. 
- On a pandas dataframe we have use to .iloc function to get the correct rows. Because I haven't split the data into X (features) and y (target) I have to also use .loc, to choose the right columns (.loc[:,features]) or simply pick the columns (['target']) for the iris dataset

![image](https://user-images.githubusercontent.com/43855029/114211785-103edd00-992f-11eb-89d0-bbd7bd0c0178.png)
```python
from sklearn.model_selection import KFold
kf10 = KFold(n_splits=10,shuffle=True,random_state=20)
for train_index, test_index in kf10.split(iris.target):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    model.fit(X_train, y_train) #Training the model, not running now
    y_pred = model.predict(X_test)
    print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test, y_pred)}")
```

## Data spliting using `Stratified K-fold`
- Instead of using random Kfold, we can use StratifiedKFold which needs extra parameter y. 
- As y you use the target variable so that the Kfold and pick balanced distribution of the targets in each folds.

```python
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
i = 1

for train_index, test_index in kf.split(iris.target):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    model.fit(X_train, y_train) #Training the model
    y_pred = model.predict(X_test)
    print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test, y_pred)}")    
```
