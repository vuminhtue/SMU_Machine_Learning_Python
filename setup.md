---
title: Setup conda environment and creating Jupyter Kernel using your SMU ManeFrame account
---

### Please follow this guideline to create a new conda environment and install scikit-learn package.

Open any terminal:
  - **MobaXTerm** for Windows OS
  - **Terminal** for MacOS & Linux platform
  - **M2 Shell Access** in Open OnDemand (https://hpc.smu.edu/pun/sys/dashboard/)

### Below is the setup using M2 Shell Access:

First, login to Open OnDemand and open Shell Access 
![image](https://user-images.githubusercontent.com/43855029/146410172-77d7531f-9673-48e7-8b3b-6e384a0893b2.png)

Request a compute node and load python library

```bash
$ srun -p htc --mem=6G --pty $SHELL
$ module load python/3
```

Create a conda environment named MyPEnv with python version 3.6

```bash
$ conda create -y -n MyPEnv python=3.7
```

Activate the conda environment and install matplotlib and any other needed packages.

```bash
[tuev@b136 ~]$ source activate myPenv
(myPenv) [tuev@b136 ~]$ conda install numpy pandas scikit-learn seaborn matplotlib -y
```

You can also create your own Jupyter Hub kernel:

```bash
(myPenv) [tuev@b136 ~]$ conda install jupyter
(myPenv) [tuev@b136 ~]$ python -m ipykernel install --user --name myPenv --display-name "My 1st conda env"
```

=> Note: while using **myPenv** conda environment, if we are missing anything, we can always come back and update using **pip install**
or **conda install** method.

- Back to Open OnDemand, request for Jupyter Notebook instance:
![image](https://user-images.githubusercontent.com/43855029/146412276-f9dd833f-2436-43cd-80b9-93b578cda2df.png)

You will see the new kernel created:

![image](https://user-images.githubusercontent.com/43855029/146412731-58cdc03b-158c-48b8-aee9-7f1f1e842efb.png)


{% include links.md %}
