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

// creating simple linear regression using numpy
m = 100
X = 2*np.random.rand(m,1) // creating columns vector
y = 4 + 3*X+np.random.randn(m,1) // creating column vector

from sklearn.preprocessing import add_dummy_feature
X_b = add_dummy_feature(X) #adding X0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T@ y // @ symbol is doing matrix multiplication

X_new = np.array([[0],[2]])
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b @ theta_best
y_predict
// Output
// array([[4.21509616],
//        [9.75532293]])
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

### Types of Gradient Descent
The project explores three main variations of Gradient Descent.
#### Batch Gradient Descent
Batch Gradient Descent calculates the gradient using the entire training set in each iteration. This makes it precise but computationally expensive for large datasets


```js
// Implementing Batch Gradient Descent
eta = 0.1 #learning rate
n_epochs = 1000
m = len(X_b) # number of instances

np.random.seed(42)
theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    gradients = 2/m*X_b.T @ (X_b @ theta - y)
    theta = theta - eta *gradients
print(theta)
// array([[4.21509616],
//        [2.77011339]])
```

The impact of different learning rates on Batch Gradient Descent is visualized.

```js
// function to plot gradient descent with different learning rates
import matplotlib as mpl

def plot_gradient_descent(theta, eta):
    m = len(X_b)
    plt.plot(X,y, "b.")
    n_epochs = 1000
    n_shown = 20
    theta_path = []
    for epoch in range(n_epochs):
        if epoch < n_shown:
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(epoch /  n_shown + 0.15))
            plt.plot(X_new, y_predict, linestyle = 'solid', color= color)
        gradients = 2/m *  X_b.T @ (X_b @ theta - y)
        theta= theta -eta * gradients
        theta_path.append(theta)
    plt.xlabel("$x_1$")
    plt.axis([0, 2, 0, 15])
    plt.grid()
    plt.title(fr"$\eta = {eta}$")
    return theta_path
```

```js
// calling the function with different eta values and plotting
np.random.seed(42)
theta = np.random.randn(2,1)

plt.figure(figsize = (10,4))
plt.subplot(131)
plot_gradient_descent(theta, eta = 0.02)
plt.ylabel("$y$", rotation=0)
plt.subplot(132)
theta_path_bgd = plot_gradient_descent(theta, eta=0.1)
plt.gca().axes.yaxis.set_ticklabels([])
plt.subplot(133)
plt.gca().axes.yaxis.set_ticklabels([])
plot_gradient_descent(theta, eta=0.5)
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735959258/gradient_descent_plot.png)

#### Stochastic Gradient Descent
Stochastic Gradient Descent (SGD) updates the parameters based on the gradient calculated from a single, randomly selected instance in the training dataset. While computationally efficient, its convergence is more erratic due to its stochastic nature.

```js
theta_path_sgd = []
n_epochs = 50
t0, t1 = 5,50  // learning schedule hyper parameters

def learning_schedule(t):
    return t0/(t + t1)

np.random.seed(42)
theta = np.random.randn(2,1)
n_shown = 20
plt.figure(figsize = (6,4))

for epoch in range(n_epochs):
    for iteration in range(m):

        if epoch == 0 and iteration < n_shown:
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(iteration / n_shown + 0.15))
            plt.plot(X_new, y_predict, color=color)

        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index +1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi ) // for SGC do not divide by m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta *gradients
        theta_path_sgd.append(theta)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
save_fig("sgd_plot")
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735959495/sgd_plot.png)

#### Mini-Batch Gradient Descent
Mini-Batch Gradient Descent strikes a balance between Batch GD and SGD. It calculates the gradient using a small, randomly selected subset (mini-batch) of the training set. This approach leverages the efficiency of matrix operations, particularly beneficial when using GPUs.

```js
from math import ceil
n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m/ minibatch_size)
np.random.seed(42)
theta = np.random.randn(2,1)
t0, t1 = 200, 1000 # learning schecule hyperparameters

def learning_schedule(t):
    return t0/(t+t1)

theta_path_mgd = []

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * minibatch_size
        xi = X_b_shuffled[idx:idx + minibatch_size]
        yi = y_shuffled[idx:idx+minibatch_size]

        gradients = 2 / minibatch_size * xi.T@(xi@theta - yi)
        eta = learning_schedule(iteration)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(7, 4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1,label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2,label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3,label="Batch")
plt.legend(loc="upper left")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$   ", rotation=0)
plt.axis([2.6, 4.6, 2.3, 3.4])
plt.grid()
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735959670/gradient_descent_paths_plot.png)

### Conclusion
This project provides a comprehensive overview of the Gradient Descent algorithm and its variants. It highlights the importance of the learning rate and the trade-offs between computational cost and convergence behavior for each type of Gradient Descent. The implementation using NumPy demonstrates the core concepts and facilitates understanding of these optimization techniques crucial for machine learning.
