---
title: Setup scikit-learn kernel in Palmetto's JupyterLab
---
We will use Palmetto cluster for this workshop with Jupyter Lab
Please follow this guideline to create a new conda environment and install scikit-learn package.
1. Open terminal (MobaXTerm for Windows OS/Terminal for MacOS & Linux platform)
2. Login to Palmetto login node: your_username@login001
3. Request for a compute node with simple configuration:

```python
$ qsub -I -l select=1:ncpus=8:mem=32gb:interconnect=any,walltime=24:00:00
```

4. Load module:

```python
$ module load anaconda3/2020.07-gcc/8.3.1
```

5. Create conda environment:

```python
$ conda create -n skln python=3.8
```

6. Once done, activate the environment and install numpy, pandas, scikit-learn, matplotlib, seaborn

```python
$ source activate skln
$ pip install numpy pandas scikit-learn seaborn
$ conda install matplotlib
```

=> Note: while using **skln** conda environment, if we are missing anything, we can always come back and update using **pip install**
or **conda install** method.

7. Last step: create Jupyter Hub kernel in order to work with Jupyter Notebook

```python
$ conda install jupyter
$ python -m ipykernel install --user --name skln --display-name "ML_SKLN"
```


8. Open Jupyter Lab in Palmetto, login and see if you have **ML_SKLN** kernel created
https://www.palmetto.clemson.edu/jhub/hub/home

![image](https://user-images.githubusercontent.com/43855029/117862252-74bbc780-b260-11eb-8dbb-4a07ae955c54.png)

{% include links.md %}
