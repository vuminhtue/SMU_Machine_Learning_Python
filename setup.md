---
title: Setup conda environment and creating Jupyter Kernel using your SMU ManeFrame account
---

There are several ways to setup the scikit-learn kernel to work:

# Method 1: Using SMU HPC M2 (Recommended)

Using this method, you need to have an M2 account.

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
(MK_SKLN) [tuev@b136 ~]$ conda install jupyter -y
(MK_SKLN) [tuev@b136 ~]$ python -m ipykernel install --user --name ML_SKLN --display-name "ML_SKLN"
```

=> Note: while using **ML_SKLN** conda environment, if we are missing anything, we can always come back and update using **pip install**
or **conda install** method.

Refresh the web browser and click on the + button on the left, you will see the new kernel created:

![image](https://user-images.githubusercontent.com/43855029/153921136-48ef26a8-3010-45a2-b17d-d3a5d5cf6805.png)

### Step 9: Make sure your installation looks ok.
Click on the ML_SKLN notebook, you will open up a Jupyter notebook. Make sure you see the ML_SKLN kernel on the top right and type in these command to check the installed version:

![image](https://user-images.githubusercontent.com/43855029/153921499-64e9cfce-46da-43a8-a0ad-9eac9fd03ca8.png)

# Method 2: Using your local machine (Windows, MacOS)

If you do not have an M2 account, not to worry, you can always use your own Laptop running on Windows/MacOS or Linux Ubuntu.

### Step 1: Download Anaconda Invididual 

Go to this [ink](https://www.anaconda.com/products/individual) and download your matched Anaconda version:

![image](https://user-images.githubusercontent.com/43855029/154314785-a13471de-1e72-4f40-b950-4c8a324e3991.png)

Once installation successful, go to step 2 to install your conda environment

### Step 2: Open Anaconda\Anaconda Navigator and Launch Jupyter Notebook

![image](https://user-images.githubusercontent.com/43855029/154315870-2bbe7811-2296-4e64-b009-e721d495c7a8.png)

You can use any web browser to open Jupyter Notebook

Right now, if you click on **New** on the right hand side, you will see some kernel appears:

![image](https://user-images.githubusercontent.com/43855029/154316152-ed4a53cd-47c4-47a8-9dab-508bcef306ec.png)

Now we gonna setup scikit-learn and create a kernel for that using **conda environment**

### Step 3: Click on Terminal:

From New, Click to Open Terminal:

![image](https://user-images.githubusercontent.com/43855029/154316322-af03a308-ee62-43b7-9971-8b037787e2c6.png)

The Terminal window appears:

![image](https://user-images.githubusercontent.com/43855029/154316407-808d311f-5a93-444d-ba2d-6d0345f99e81.png)

### Step 4: Create a conda environment named ML_SKLN with python version 3.6

```bash
PS C:\Users\46791130> conda create -y -n ML_SKLN python=3.6
```

If you see this screen, it means your conda environment was created and you are ready to install scikit-learn:

![image](https://user-images.githubusercontent.com/43855029/154316699-e2a356b2-f90e-4563-9a1d-8e1879f34607.png)

### Step 5: Activate the conda environment 

For MacOS:

```bash
PS C:\Users\46791130> source activate ML_SKLN
```

For Windows:
Go back to Anaconda Navigator. Click on Environment, you will see the ML_SKLN environment was created. Click to Open Terminal:

![image](https://user-images.githubusercontent.com/43855029/154318048-45c98550-019d-4d9f-b04b-ed1388d20dc2.png)

### Step 6: Once the ML_SKLN conda environment was activated,install scikit-learn, matplotlib and any other needed packages

```bash
(ML_SKLN) C:\Users\46791130> conda install numpy pandas scikit-learn seaborn matplotlib -y
```

### Step 7: Install jupyter and create ML_SKLN kernel:

```bash
(ML_SKLN) C:\Users\46791130> conda install jupyter -y
(ML_SKLN) C:\Users\46791130> python -m ipykernel install --user --name ML_SKLN --display-name "ML_SKLN"
```

=> Note: while using **ML_SKLN** conda environment, if we are missing anything, we can always come back and update using **pip install**
or **conda install** method.

Once installation done, go back to Jupyter Notebook and hit refresh. Click on New and you will see ML_SKLN kernel created:

![image](https://user-images.githubusercontent.com/43855029/154318863-a1a9daef-6498-4b19-9bcd-69d9b2eab8b0.png)

Click on ML_SKLN kernel and a Jupyter Notebook appears with ML_SKLN kernel on the top right:

![image](https://user-images.githubusercontent.com/43855029/154319064-a79b3b8b-2167-47b0-8aa5-150fb49e6f00.png)

You are good to go!
{% include links.md %}
