---
title: Setup conda environment and creating Jupyter Kernel using your SMU ManeFrame account
---
### Step 1: Login to SMU Open OnDemand via web browser:

```
hpc.smu.edu
```

![image](https://user-images.githubusercontent.com/43855029/153682287-97f59016-5b64-49c6-bfd2-a12dd0861a55.png)

### Step 2: Request JupyterLab Instance:

![image](https://user-images.githubusercontent.com/43855029/153682171-28f85d28-086c-4c25-9f58-987d9c300728.png)

### Step 3: Fill in the following information and hit Launch:

![image](https://user-images.githubusercontent.com/43855029/153682400-1eb87cd9-91f3-4177-b63e-100d49fa77d9.png)

### Step 4: Connect to Jupyter Lab when it is ready:

![image](https://user-images.githubusercontent.com/43855029/153682468-7f759b33-b246-4bb5-801f-f2d36fad76dd.png)

### Step 5: Click on Terminal:

![image](https://user-images.githubusercontent.com/43855029/153682514-b89dcd3b-866e-4782-94e7-d61ac2b1b492.png)

### Step 6: Create a conda environment named ML_SKLN with python version 3.6

```bash
[tuev@b136 ~]$ conda create -y -n ML_SKLN python=3.6
```

### Step 7: Activate the conda environment and install scikit-learn, matplotlib and any other needed packages.

```bash
[tuev@b136 ~]$ source activate ML_SKLN
(MK_SKLN) [tuev@b136 ~]$ conda install numpy pandas scikit-learn seaborn matplotlib -y
```

### Step 8: Install jupyter and create ML_SKLN kernel:

```bash
(MK_SKLN) [tuev@b136 ~]$ conda install jupyter
(MK_SKLN) [tuev@b136 ~]$ python -m ipykernel install --user --name ML_SKLN --display-name "ML_SKLN"
```

=> Note: while using **ML_SKLN** conda environment, if we are missing anything, we can always come back and update using **pip install**
or **conda install** method.

You will see the new kernel created:

![image](https://user-images.githubusercontent.com/43855029/146412731-58cdc03b-158c-48b8-aee9-7f1f1e842efb.png)


{% include links.md %}
