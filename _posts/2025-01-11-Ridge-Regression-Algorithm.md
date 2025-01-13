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

#### Adding Intercept Term
An intercept term is added to the predictor variables to represent the baseline prediction when all other features are zero. This is crucial for accurate model fitting.
```js
X["intercept"] = 1
X = X[["intercept"] + predictors]
```

#### Ridge Regression Implementation
The core of the project involves implementing the Ridge Regression equation from scratch. This includes creating the penalty matrix using the chosen alpha (regularization strength) and the identity matrix, and then calculating the coefficients (B) using the formula:

**B = (XᵀX + αI)⁻¹Xᵀy**

```js
alpha = 2
I = np.identity(X.shape[1])
I[0][0] = 0
penalty = alpha * I
B = np.linalg.inv(X.T @ X + penalty) @ X.T @ y
```

The calculated coefficients represent the weights assigned to each predictor variable.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1736740780/manual_ridge.png)

#### Prediction and Evaluation

The trained model is then used to predict the number of medals for the test set.

```js
test_X = test[predictors]
test_X = (test_X - x_mean) / x_std
test_X["intercept"] = 1
test_X = test_X[["intercept"] + predictors]
predictions = test_X @ B
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1736740959/Ridge_manual_predictions.png)

#### Comparison with Scikit-learn
The project compares the results of the manual implementation with the Scikit-learn Ridge Regression model. This demonstrates the consistency and validity of the manual implementation.

```js
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = alpha)
ridge.fit(X[predictors],y)
ridge.coef_
// Output array([[ 61.85773366, -34.63292036]])
ridge.intercept_
// Output array([10.69149597])
```

Manual predictions - sklearn_predictions
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1736744369/manual_sklearn_comparison_ridge.png)


#### Hyperparameter Tuning (Alpha)
The impact of the regularization strength (alpha) on model performance is explored by training the model with different alpha values and comparing the resulting MAE.

```js
from sklearn.metrics import mean_absolute_error

errors = []
alphas = [10**i for i in range(-2,4)]

for alpha in alphas:
    B, x_mean, x_std = ridge_fit(train, predictors, target, alpha)
    predictions = ridge_predict(test, predictors, x_mean, x_std, B)
    errors.append(mean_absolute_error(test[target], predictions))

// alphas [0.01, 0.1, 1, 10, 100, 1000]
```

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1736744602/ridge_errors.png)
This analysis reveals the optimal alpha value that minimizes the prediction error.

#### Conclusion
This project successfully demonstrates the application of Ridge Regression for predicting Olympic medal counts. The manual implementation provides a deep understanding of the underlying mathematical concepts, while the comparison with Scikit-learn validates the approach. The hyperparameter tuning process showcases the importance of finding the right balance between model complexity and predictive accuracy. Future work could explore incorporating additional features and experimenting with other regularization techniques to further improve the model's performance.
