---
title: Detailed explanation on Decision Tree
---

# Decision Tree

Splitting algorithm

- Gini Impurity: (Categorical)
- Chi-Square index (Categorical)
- Cross-Entropy & Information gain (Categorical)
- Reduction Variance (Continuous)

## Example on Decision Tree 
Assume we have a sample data of 30 students in the training set with three input variables Gender (Boy/ Girl), Class( IX/ X) and Height (5 to 6 ft); one output variable: Play_Cricket (binary)
 - 15 out of these 30 play cricket in leisure time (15 Yes and 15 No)
 - predict who will play cricket during leisure period?

![image](https://user-images.githubusercontent.com/43855029/121087243-30194280-c7b2-11eb-9832-7df802385977.png)

This is a typical problem for Decision Tree algorithm

Next we gonna use different splitting algorithm to split the nodes:

### Gini Impurity
![image](https://user-images.githubusercontent.com/43855029/121087017-e4669900-c7b1-11eb-9e6f-c0b47f1e0afb.png)

As there are 3 input variables (Gender, Class, Height), the algorithm will be splitting for all variables and calculate the Gini Impurity correspondingly based on the above formulation:

#### Splitting based on Gender:

![image](https://user-images.githubusercontent.com/43855029/121087345-4e7f3e00-c7b2-11eb-978f-790b9223a8c9.png)

- The whole population of splitting is 30, the probability of Yes & No are 50% for the whole sample
- After splitting by Gender, the number of Male students is 20, the number of cricket player is 13, equivalent to 65% male population. Therefore Gini value of Male is: ![image](https://user-images.githubusercontent.com/43855029/121094477-b5095980-c7bc-11eb-8db4-45c572b87691.png)

- the number of Female students is 10, the number of cricket player is 2, equivalent to 20% female population. Therefore Gini value of Female is: ![image](https://user-images.githubusercontent.com/43855029/121094541-cfdbce00-c7bc-11eb-803f-493a141718f5.png)

- The Gini Impurity for Gender splitting is: ![image](https://user-images.githubusercontent.com/43855029/121094621-ee41c980-c7bc-11eb-9c43-7d6c50124c13.png)

#### Similarly, splitting based on Height:

![image](https://user-images.githubusercontent.com/43855029/121094742-17faf080-c7bd-11eb-8131-dc7a16a6f00a.png)

- The Gini Impurity for Height splitting is 0.5

#### Splitting based on Class:

![image](https://user-images.githubusercontent.com/43855029/121094840-3eb92700-c7bd-11eb-925b-d34cd37130a6.png)

- The Gini Impurity for Class splitting is 0.49

Base on the 3 Gini Impurity on the splitting, we go with the smallest values, which is splitting by Gender.

The process is continue with the next nodes.

### Chi-Squared

![image](https://user-images.githubusercontent.com/43855029/121095363-37464d80-c7be-11eb-87bc-dd2fa3093c2c.png)

### Entropy

![image](https://user-images.githubusercontent.com/43855029/121095141-c56e0400-c7bd-11eb-91ce-f8da11ad66aa.png)




{% include links.md %}
