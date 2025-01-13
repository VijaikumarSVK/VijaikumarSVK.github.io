---
date: 2025-01-12 12:26:40
layout: post
title: Ridge Regression Algorithm
subtitle: Predicting Olympic Medals with Ridge Regression
description: Predicting Olympic medal counts using Ridge Regression, exploring data analysis and regularization techniques.
image: https://res.cloudinary.com/dqqjik4em/image/upload/v1736739086/Ridge-Regression.png
optimized_image: https://res.cloudinary.com/dqqjik4em/image/upload/f_auto,q_auto/Ridge-Regression
category: Data Science
tags:
  - ML
  - Optimization
  - Algorithm
author: Vijai Kumar
vj_layout: false
vj_side_layout: true
---

**Executive Summary:** TThis project explores the application of Ridge Regression, a regularized linear regression technique, to predict the number of medals a country will win at the Olympic Games. Using historical data on participating teams, including the number of athletes, events participated in, age, height, weight, and previous medal counts, the model aims to identify the key factors contributing to Olympic success. The project demonstrates the entire process, from data preprocessing and model training to evaluation and comparison with the Scikit-learn implementation of Ridge Regression. The analysis reveals the importance of regularization in preventing overfitting and highlights the trade-off between model complexity and predictive accuracy.

The complete Python code and required file for this analysis is available on my <b><a href="https://github.com/VijaikumarSVK/Ridge-Regression">GitHub</a></b>

### Introduction to Gradient Descent
Predicting the success of Olympic teams is a complex task involving numerous factors. This project utilizes Ridge Regression, a powerful statistical method, to tackle this challenge. Ridge Regression addresses the problem of multicollinearity, where predictor variables are highly correlated, by adding a penalty term to the ordinary least squares objective function. This regularization helps prevent overfitting, leading to a more robust and generalizable model.

#### Data Loading and Preprocessing
The project begins by loading the Olympic teams' data from a CSV file using pandas. The dataset contains information on various teams, including the year of participation, number of athletes, events participated in, average age, height, weight, and previous medal counts.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1736739873/olympic_data.png)

A large learning rate might cause the algorithm to overshoot the minimum, potentially leading to divergence.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735957739/larger_gradient_value.png)

The dataset is then split into training and testing sets to evaluate the model's performance on unseen data.

```js
train, test = train_test_split(teams, test_size=0.2, random_state=1)
```

#### Feature Selection and Target Variable
For this project, the number of athletes and events participated in are selected as predictor variables (features), while the number of medals won serves as the target variable.

```js
predictors = ["athletes", "events"]
target = "medals"
X = train[predictors].copy()
y = train[[target]].copy()
```
#### Data Scaling
To ensure that all features contribute equally to the model, the predictor variables are standardized by subtracting the mean and dividing by the standard deviation. This prevents features with larger values from dominating the model.
```js
x_mean = X.mean()
x_std = X.std()
X = (X - x_mean) / x_std

// We rescaled our STD to 1 and mean to Zero
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1736740203/Ridge_scaled_data.png)

<!--
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
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735959824/sgd_plot.png)

#### Mini-Batch Gradient Descent
Mini-Batch Gradient Descent strikes a balance between Batch GD and SGD. It calculates the gradient using a small, randomly selected subset (mini-batch) of the training set. This approach leverages the efficiency of matrix operations, particularly beneficial when using GPUs.

```js
from math import ceil
n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m/ minibatch_size)
np.random.seed(42)
theta = np.random.randn(2,1)
t0, t1 = 200, 1000 // learning schecule hyperparameters

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
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1735959830/gradient_descent_paths_plot.png)

### Conclusion
This project provides a comprehensive overview of the Gradient Descent algorithm and its variants. It highlights the importance of the learning rate and the trade-offs between computational cost and convergence behavior for each type of Gradient Descent. The implementation using NumPy demonstrates the core concepts and facilitates understanding of these optimization techniques crucial for machine learning. -->
