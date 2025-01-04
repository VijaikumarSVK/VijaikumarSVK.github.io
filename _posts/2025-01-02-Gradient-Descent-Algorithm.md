---
date: 2025-01-02 12:26:40
layout: post
title: Gradient Descent Algorithm
subtitle: Optimizing ML Model Performance with Different Gradient Descent Methods
description: A breakdown of essential machine learning optimization algorithm.
image: https://res.cloudinary.com/dqqjik4em/image/upload/v1735925954/random%20initial%20value%20-%20cover%20page.png
optimized_image: https://res.cloudinary.com/dqqjik4em/image/upload/f_auto,q_auto/random%20initial%20value%20-%20cover%20page
category: Data Science
tags:
  - ML
  - Optimization
  - Algorithm
author: Vijai Kumar
vj_layout: false
vj_side_layout: true
---

**Executive Summary:** This project explores the Gradient Descent algorithm, a fundamental optimization technique in machine learning used to find the optimal parameters of a model. The project covers different types of Gradient Descent, including Batch Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent, comparing their performance and characteristics. The implementation demonstrates how these algorithms iteratively adjust model parameters to minimize a cost function, illustrated with a simple linear regression example. The visualizations highlight the convergence behavior and the impact of the learning rate on each algorithm.

The complete Python code and required file for this analysis is available on my <b><a href="https://github.com/VijaikumarSVK/Gradient-Descent-Algorithm">GitHub</a></b>

### Introduction to Gradient Descent
Gradient Descent is a powerful **optimization algorithm** widely used in machine learning to find the optimal parameters of a model. It iteratively adjusts the parameters to minimize a cost function, which measures the difference between the model's predictions and the actual target values. The algorithm mimics the process of descending a hill, taking steps in the direction of the steepest descent to reach the lowest point.

The process begins by initializing the model parameters with random values. Then, the algorithm calculates the gradient of the cost function with respect to each parameter. The gradient indicates the direction of the steepest ascent, so the algorithm updates the parameters by taking a step in the opposite direction (steepest descent). The size of this step is controlled by the learning rate, a crucial hyperparameter that influences the algorithm's convergence.


#### The Learning Rate
The learning rate determines the size of the steps taken during gradient descent. A small learning rate leads to slow convergence, requiring many iterations to reach the minimum.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735957419/small%20gradient%20value.png)

A large learning rate might cause the algorithm to overshoot the minimum, potentially leading to divergence.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735957739/larger_gradient_value.png)

Gradient descent, while effective, isn't fool proof. The shape of the cost function can pose challenges. "Irregular terrain" like holes or ridges can prevent finding the absolute best solution (global minimum), sometimes trapping the algorithm in a suboptimal solution (local minimum). Flat areas (plateaus) can also slow progress significantly, requiring extensive computation to traverse. A poor starting point can exacerbate these problems, leading to either a local minimum or slow convergence.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735957857/gradient_pitfall.png)


### Implementation: Simple Linear Regression
The project demonstrates Gradient Descent with a simple linear regression example using NumPy. First, we generate synthetic data for the linear regression.


```js
import numpy as np
import matplotlib.pyplot as plt

# creating simple linear regression using numpy
m = 100
X = 2*np.random.rand(m,1) # creating columns vector
y = 4 + 3*X+np.random.randn(m,1) # creating column vector

from sklearn.preprocessing import add_dummy_feature
X_b = add_dummy_feature(X) #adding X0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T@ y # @ symbol is doing matrix multiplication

X_new = np.array([[0],[2]])
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b @ theta_best

#plotting Linear Regression chart
plt.figure(figsize = (6,4))
plt.plot(X_new, y_predict, 'r-', label = 'Predictions')
plt.plot(X, y, 'b.')
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation = 0)
plt.axis([0,2,0,15])
plt.legend(loc = 'upper left')
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735957857/linear_model_predictions_plot.png)















<!-- ttt -->
