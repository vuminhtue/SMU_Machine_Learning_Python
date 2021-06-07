---
title: Detailed explanation on Decision Tree
---

# Decision Tree
## Splitting algorithm

- Gini Impurity: (Categorical)
- Chi-Square index (Categorical)
- Cross-Entropy & Information gain (Categorical)
- Reduction Variance (Continuous)

### Example on Decision Tree 
Assume we have a sample data of 30 students in the training set with three input variables Gender (Boy/ Girl), Class( IX/ X) and Height (5 to 6 ft); one output variable: Play_Cricket (binary)
 - 15 out of these 30 play cricket in leisure time (15 Yes and 15 No)
 - predict who will play cricket during leisure period?

![image](https://user-images.githubusercontent.com/43855029/121087243-30194280-c7b2-11eb-9832-7df802385977.png)

This is a typical problem for Decision Tree algorithm

Next we gonna use different splitting algorithm to split the nodes:

#### Gini Impurity
![image](https://user-images.githubusercontent.com/43855029/121087017-e4669900-c7b1-11eb-9e6f-c0b47f1e0afb.png)

As there are 3 input variables (Gender, Class, Height), the algorithm will be splitting for all variables and calculate the Gini Impurity correspondingly based on the above formulation:

- Splitting based on Gender:

![image](https://user-images.githubusercontent.com/43855029/121087345-4e7f3e00-c7b2-11eb-978f-790b9223a8c9.png)

- The whole population of splitting is 30, the probability of Yes & No are 50% for the whole sample
- After splitting by Gender, the number of Male students is 20, the number of cricket player is 13, equivalent to 65% male population
- the number of Female students is 10, the number of cricket player is 2, equivalent to 20% female population.
- The Gini Impurity for Gender spliting is:
- 
- 

{% include links.md %}
