---
title: "Support Vector Machine"
teaching: 20
exercises: 0
questions:
- "How to use Support Vector Machine in Machine Learning model"
objectives:
- "Learn how to use SVM in ML model"
keypoints:
- "SVM"
---

## Support Vector Machine
The objective of the support vector machine (SVM) algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points.

### Applications of Support Vector Machine:
![image](https://user-images.githubusercontent.com/43855029/114576381-1394da00-9c49-11eb-95b1-cff9d87c6029.png)

### Explanation
- To separate the two classes of data points, there are many possible hyperplanes that could be chosen

![image](https://user-images.githubusercontent.com/43855029/114577032-af264a80-9c49-11eb-8e6c-b45120743f0d.png)

- SVM's objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes.
Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

![image](https://user-images.githubusercontent.com/43855029/114576981-a2a1f200-9c49-11eb-9921-b0bff879c97e.png)

- Example of hyperplane in 2D and 3D position:

![image](https://user-images.githubusercontent.com/43855029/114577340-eac11480-9c49-11eb-8ff9-4aa3e61b1c86.png)

- Support vectors (**SVs**) are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane.
Using **SVs** to maximize the margin of the classifier.
Removing **SVs** will change the position of the hyperplane. These are the points that help us build our SVM.

![image](https://user-images.githubusercontent.com/43855029/114577489-09271000-9c4a-11eb-8b4a-b7837463288f.png)

### Support Vector Machines's Estimators:
There are 2 main types of SVM in sklearn, depending on the model output:

- **SVC**: for Classification problem using **C-support vector classification**
- **SVR**: for Regression problem using **Epsilon-support vector regression**

In addition there are other model under **sklearn.svm**: NuSVC, LinearSVC, NuSVR, LinearSVR



### Kernel function:
![image](https://user-images.githubusercontent.com/43855029/115589944-6cdeb800-a29e-11eb-858b-ff278bb56a3d.png)

### Implementation
Here we use the regular **iris** dataset with Classification problem

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
iris = load_iris()
X = iris.data
y = pd.DataFrame(iris.target)
y['Species']=pd.Categorical.from_codes(iris.target, iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.6,random_state=123)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Fit Support Vector Classifier model
```python
from sklearn.svm import SVC
model_svm = SVC(kernel='rbf', C=1).fit(X_train, y_train['Species'])
model_svm.score(X_test,y_test['Species'])
```
In this model, **C** is the regularization parameter `Default C=1`. The strength of the regularization is inversely proportional to C. Must be strictly positive.

### Tips on using SVM
- Setting `C=1` is reasonable choice for default. If you have a lot of noisy observations you should decrease it: decreasing C corresponds to more regularization.
- Support Vector Machine algorithms are not scale invariant, so it is highly recommended to **scale your data**. 
- More information [here](https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use)
