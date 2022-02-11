---
title: "Training Supervised Machine Learning model with Categorical Output"
teaching: 20
exercises: 0
questions:
- "How to train a Machine Learning model with Categorical output?"
objectives:
- "Learn different ML Supervised Learning with Categorical output"
keypoints:
- "Decision Tree, Random Forest"
---

# 6 Supervised Learning with categorical output

- Typical Classification problem with 2, 3, 4 (or more) outputs.
- Most of the time the output consists of binary (male/female, spam/nospam,yes/no) 
- Sometime, there are more than binary output: dog/cat/mouse, red/green/yellow.

In this category, we gonna use 2 existing dataset from [sklearn](https://scikit-learn.org/stable/datasets.html):
- [Breast Cancer Wisconsine](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset) data for Binary output
- [Iris plant](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) data for multiple (3) output.

## 6.1 Logistic Regression for binary output

- Logistic regression is another technique borrowed by machine learning from the field of statistics. It is the go-to method for binary classification problems (problems with two class values).
- Typical binary classification: True/False, Yes/No, Pass/Fail, Spam/No Spam, Male/Female
- Unlike linear regression, the prediction for the output is transformed using a non-linear function called the logistic function.
- The standard logistic function has formulation:

![image](https://user-images.githubusercontent.com/43855029/114233181-f7dcbb80-994a-11eb-9c89-58d7802d6b49.png)

![image](https://user-images.githubusercontent.com/43855029/114233189-fb704280-994a-11eb-9019-8355f5337b37.png)

In this example, we load a sample dataset called [Breast Cancer Wisconsine](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset).

### Load Breast Cancer Wisconsine data

```python
from sklearn.datasets import load_breast_cancer

# generate sample data
X = data.data
y = data.target
print("There are", X.shape[1], " Predictors: ", data.feature_names)
print("The output has 2 values: ", data.target_names)
print("Total size of data is ", X.shape[0], " rows")
```

We can see that there are 30 input data representing the shape and size of 569 tumours.
Base on that, the tumour can be considered _malignant_ or _benign_ (0 or 1 as in number)

### Partitioning Data to train/test:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=123)
```

### Train model using Logistic Regression
For simplicity, we use all predictors for the regression:

```python
from sklearn.linear_model import LogisticRegression
model_LogReg = LogisticRegression(solver='newton-cg').fit(X_train, y_train)

### Evaluate model output:

```python
y_pred = model_LogReg.predict(X_test)

from sklearn import metrics
print("The accuracy score is %1.3f" % metrics.accuracy_score(y_test,y_pred))
```

We retrieve the **accuracy = 0.965** using all predictors

### Compute AUC-ROC and plot curve

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

lr_probs = model_LogReg.predict_proba(X_test)
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

![image](https://user-images.githubusercontent.com/43855029/153662934-d4c5929f-72cf-43b8-8b1f-085d315022e7.png)

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

![image](https://user-images.githubusercontent.com/43855029/153663219-f27aad2b-b76d-4abf-a093-0a433e79bd28.png)


## 6.2 Classification problem with more than 3 outputs

Here we use [Iris plant](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) data for multiple (3) output.

### Import data

```python
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

print("There are", X.shape[1], " Predictors: ", data.feature_names)
print("The output has 3 values: ", data.target_names)
print("Total size of data is ", X.shape[0], " rows")
```

- We can see that there are 4 input data representing the petal/sepal width and length of 3 different kind of iris flowers.
- Base on that, the iris plants can be classified as 'setosa' 'versicolor' 'virginica'.

### Partitioning Data to train/test:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=123)
```

### Train model using Linear Discriminant Analysis (LDA):

For simplicity, we use all predictors for the regression:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model_LDA = LinearDiscriminantAnalysis().fit(X_train,y_train)
```

### Evaluate model output:

```python
print("The accuracy score is %1.3f" % model_LDA.score(X_test,y_test))
```

### LDA can be used for both binary and more categorical output

Exercise: create an LDA model to predict the breast cancer Wisconsine data

```python

```

## 6.3 Other Algorithms

There are many other algorithms that work well for both classification and regression data such as Decision Tree, RandomForest, Bagging/Boosting.
Very similar to chapter 5, the following model should be loaded:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
```

Exercise: create a Random Forest model to predict the iris flower data using the same method:

```python

```
