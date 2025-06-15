# Restaurant Profit Prediction using Linear Regression
This repository contains a practice lab from Course 1: Supervised Machine Learning – Regression and Classification, part of the Machine Learning Specialization by Andrew Ng on Coursera.

The lab focuses on Linear Regression with one variable using Gradient Descent to predict restaurant profits based on city population data.

## Problem Statement
You are the CEO of a restaurant chain planning to expand into new cities. You have data showing the population of various cities along with the profits made by your restaurants in those cities. Your goal is to:

* Predict the expected profit of a restaurant in a new city based on its population.
* Build a linear regression model to learn the relationship between city population and profit.
* Use gradient descent to find the best model parameters that minimize prediction error.
* Visualize the results and evaluate the model's performance.

## Libraries Used
```python

import numpy as np
import matplotlib.pyplot as plt
from utils import *
```

* numpy is the fundamental package for working with matrices in Python.
* matplotlib is a famous library to plot graphs in Python.
* utils.py contains helper functions for this assignment. You do not need to modify code in this file.


## Data Exploration and Visualization
  
The load_data() function shown below loads the data into variables x_train and y_train
```python

x_train, y_train = load_data()
```

Before building the model, it is important to understand the dataset.

### View Variables:
The features (x_train) and labels (y_train) are printed to get a sense of their contents and types.
* x_train is a NumPy array representing city populations (in units of 10,000). For example, a value of 6.1101 corresponds to a city population of 61,101.
* y_train is a NumPy array representing monthly restaurant profits (in $10,000 units). Values can be positive (profit) or negative (loss).

### Check Dimensions:
Both x_train and y_train have 97 data points, indicating 97 training examples.

### Visualize Data:
A scatter plot is created to visualize the relationship between city population and restaurant profit. Each point represents one city.
This helps to visually confirm if there appears to be a linear relationship, which is important before applying linear regression.
```python

# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()
```
![image](https://github.com/user-attachments/assets/308ec7e2-3db2-4d4c-b188-6aacef22ac06)

## Goal: 
The goal is to build a linear regression model to fit this data. With this model, you can then input a new city's population, and have the model estimate your restaurant's potential monthly profits for that city.

# Compute Cost Function 
![image](https://github.com/user-attachments/assets/52fd715e-7c88-4830-9271-c939cd6e9c4e)
![image](https://github.com/user-attachments/assets/dc91cd4b-72db-467f-9514-d1dee872ff05)


```python
def compute_cost(x, y, w, b): 
    m = x.shape[0]
    cost_sum = 0
  for i in range(m):
    f_wb = w * x[i] + b
    cost_sum += (f_wb - y[i]) ** 2

total_cost = (1 / (2 * m)) * cost_sum
return total_cost
```

# Compute Gradient Function

![image](https://github.com/user-attachments/assets/66662804-fe5d-4fb6-b1c4-138747e4b362)
![image](https://github.com/user-attachments/assets/166deef6-a4fb-44bf-9102-6afd47a18a61)

```python
def compute_gradient(x, y, w, b): 
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
```

## Gradient Descent Optimization
Parameters w and b were updated iteratively using the computed gradients to minimize the cost function, leading to the best-fit linear regression model.

## Repository Structure
restaurant-profit-prediction/

├── utils.py                # Helper functions like data loading

├── public_tests.py # Automated test functions to verify the correctness of your core model functions

├── linear_regression.ipynb # Main notebook with implementation and results

├── README.md               # This document

## Getting Started
To run this notebook:
1. Clone the repository
```python
git clone https://github.com/GhadeerZahwe/restaurant-profit-prediction.git
```
2. Install required libraries (numpy, matplotlib)
3. Open and run linear_regression.ipynb in Jupyter Notebook or Google Colab

## Note
This code is implemented for educational purposes following the Coursera Machine Learning course guidelines.
