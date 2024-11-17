---
date: 2024-08-29 12:26:40
layout: post
title: Classification-Handwritten Digits
subtitle: A Comparative Study of Digit Classification Algorithms
description: KNN/SGD achieves 97% on MNIST dataset. Multilabel/output classification explored
image: https://res.cloudinary.com/dqqjik4em/image/upload/v1731812902/classification_cover_page_2.jpg
optimized_image: https://res.cloudinary.com/dqqjik4em/image/upload/f_auto,q_auto/classification_cover_page_2
category: Data Science
tags:
  - ML
  - Classification
  - Model Evaluation
author: Vijai Kumar
vj_layout: false
vj_side_layout: true
---

**Executive Summary:** This project aims to classify handwritten digits (0-9) using machine learning techniques, primarily focusing on the MNIST(Modified National Institute of Standards and Technology database) dataset. The project explores various classification algorithms, evaluates their performance using metrics like accuracy, precision, recall, and F1-score, and implements techniques like cross-validation and hyperparameter tuning to optimize model performance. It also delves into more advanced concepts such as multilabel and multioutput classification.

The complete Python code and required file for this analysis is available on my <b><a href="https://github.com/VijaikumarSVK/Classification-Handwritten-Digits">GitHub</a></b>

### Data and Initial Setup
#### MNIST Dataset Introduction
The **MNIST** dataset, a cornerstone in image recognition, comprises 70,000 grayscale images of handwritten digits (0-9), each measuring 28x28 pixels. These images, contributed by high school students and US Census Bureau employees, serve as a robust benchmark for evaluating classification models. Each image is labeled with the digit it represents.

```js
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame = False)
```
Data exploration reveals the dataset's structure: **70,000 samples,** each with **784 features** (representing the pixel values) and corresponding labels.<br>

Below sample image represents digit 5
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731814161/some_digit_plot.png)

Visualizing a sample image helps understand the data format and verify the label accuracy.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731814490/more_digits_plot.png)
Displaying a grid of images provides a comprehensive overview of the dataset's diversity.

#### Binary Classification with SGD
As a starting point, the project simplifies the classification problem to a binary classification task: identifying the digit "5". This creates a "5-detector" that distinguishes between two classes: "5" and "not-5". The Stochastic Gradient Descent (SGD) classifier, well-suited for large datasets, is employed for this task.

```js
y_train_5 = (y_train =='5') // true for all 5's, False for all other digits
y_test_5 = (y_test == '5')
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit]) // Output -> array([ True])
```

### Model Evaluation and Performance
#### Cross-Validation for Accuracy Measurement
Cross-validation, a robust evaluation technique, is used to assess the model's performance. It involves partitioning the training set into folds, training the model on different combinations of these folds, and evaluating its performance on the held-out fold. This provides a more reliable estimate of the model's generalization ability.


```js
//cross-validation and calculate accuracy
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = 'accuracy')
// Output -> array([0.95035, 0.96035, 0.9604 ])

//cross-validation manually
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits = 3)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
//Output
// 0.95035
// 0.96035
// 0.9604
```

#### Confusion Matrix Analysis
Confusion matrices offer a detailed breakdown of model predictions, highlighting the counts of true positives, true negatives, false positives, and false negatives. They provide valuable insights into the types of errors the model is making.

```js
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)
// Output
// array([[53892,   687],
//        [ 1891,  3530]], dtype=int64)
```
 The first row of this matrix considers non-5 images (the negative class): 53,892 of them were correctly classified as non-5s (they are called **true negatives**), while the remaining 687 were wrongly classified as 5s (**false positives**, also called type I errors). The second row considers the images of 5s (the positive class): 1,891 were wrongly classified as non-5s (**false negatives**, also called type II errors), while the remaining 3,530 were correctly classified as 5s (**true positives**).

#### Precision and Recall Metrics
Precision and recall are crucial metrics for evaluating a classifier's performance, especially in imbalanced datasets. Precision measures the accuracy of positive predictions, while recall quantifies the model's ability to identify all positive instances.
```js
//Precision score
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) == 3530 / (687 + 3530)
// Output -> 0.8370879772350012
```
```js
//Recall score
recall_score(y_train_5, y_train_pred)# == 3530 / (1891 + 3530)
// Output -> 0.6511713705958311
```
The F1-score, the harmonic mean of precision and recall, provides a single metric that balances both aspects of performance.<br>

**F1 = 2 × ((precision × recall)/(precision + recall))**

```js
from sklearn.metrics import f1_score
f1_score(y_train_5,y_train_pred)
Output -> 0.7325171197343846
```

#### Precision-Recall Trade-off
There's often an inverse relationship between precision and recall. Increasing the prediction threshold can improve precision at the expense of recall, and vice-versa. Analyzing this trade-off is essential for selecting an optimal threshold based on the application's specific requirements.
```js
threshold = 3000
y_some_digit_pred = (y_scores>threshold)
y_some_digit_pred
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```
```js
plt.figure(figsize=(8,4))
plt.plot(thresholds, precisions[:-1], "b--", label = 'Precision', linewidth = 2)
plt.plot(thresholds, recalls[:-1], "g-", label  ='Recall', linewidth = 2)
plt.vlines(threshold, 0,1.0,"k", "dotted", label = "threshold")

idx = (thresholds >= threshold).argmax()
plt.plot(thresholds[idx], precisions[idx],'bo')
plt.plot(thresholds[idx], recalls[idx],"go")
plt.axis([-50000,50000,0,1])
plt.grid()
plt.xlabel('Threshold')
plt.legend(loc = "center right")
save_fig("precision_recall_vs_threshold_plot")
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731816754/precision_recall_vs_threshold_plot.png)

Another way to select a good precision/recall trade-off is to plot precision directly against recall
```js
import matplotlib.patches as patches
plt.figure(figsize=(6,5))
plt.plot(recalls, precisions, linewidth = 2, label = 'Precision/Recall curve')
plt.plot([recalls[idx], recalls[idx]],[0.,precisions[idx]], "k:")
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
plt.plot([recalls[idx]], [precisions[idx]], "ko", label = "Point at threshold 3,000")
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.79, 0.60), (0.61, 0.78),
    connectionstyle="arc3,rad=.2",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color = "#333333")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0,1,0,1])
plt.grid()
plt.legend(loc = 'lower left')
save_fig("precision_vs_recall_plot")
plt.show()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1731817186/precision_vs_recall_plot.png)


```js
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision] // --> 3370.0194991439557
y_train_pred_90 = (y_scores >= threshold_for_90_precision)
precision_score(y_train_5, y_train_pred_90) // --> 0.9000345901072293
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90) // --> 0.4799852425751706
```
We have 90% precision classifier, but high precision classifier is not very usefull if its recall is too low. for many application 48% recall wouldn't be great at all










```js

```

<!-- ################################## -->



#### Data Visualization
Visualizing the data is crucial for understanding its underlying patterns. We create histograms to examine the distribution of each numerical attribute. This helps us identify potential issues like skewed distributions or capped values, which may impact model performance.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729822514/attribute_histogram_plots.png)

The histograms reveal interesting insights:
1. **Median income:** The data is capped, with the highest value at 15, representing $150,000. This might affect the model's ability to predict prices accurately for high-income areas.
2. **Housing median value:** The housing prices are also capped, possibly limiting the model's prediction range.
3. **Skewed Distributions:** Many histograms exhibit right-skewness, potentially challenging certain machine learning algorithms in detecting patterns.

#### Creating a Stratified Train-Test Split
To avoid data snooping bias, we create a stratified train-test split based on the **median_income** attribute. This ensures that both the training and testing sets have representative proportions of different income categories, improving the model's generalizability.

```js
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits = 10 , test_size = 0.2, random_state = 42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing['income_cat']):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_test_set_n, strat_train_set_n])
```
We compare the income category proportions in the overall dataset, stratified test set, and random test set. The stratified split maintains a near-perfect representation of the original proportions, minimizing potential bias

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729824366/compare_table.png)

### Exploratory Data Analysis (EDA)
Now, we dive deeper into the training data to explore relationships between attributes and identify potential patterns.

#### Visualizing Geographical Data
We begin by visualizing the geographical distribution of housing data using scatter plots. This provides a visual representation of housing density and price variations across different regions.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729824996/Geographical_scatter_plot.png)

The plot highlights areas with high housing density, like the Bay Area, Los Angeles, and San Diego, suggesting potential price premiums in these locations.

##### Housing Price as per Location and Population
We enhance the scatter plot by incorporating housing prices and population density. This allows us to observe correlations between these factors and geographical location.
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729826728/housing_prices_scatterplot.png)

The plot reveals a strong correlation between housing prices and location, with coastal areas exhibiting higher prices. Population density also seems to play a role, with denser areas generally having higher housing costs.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729826867/california_housing_prices_plot.png)

This visualization reinforces the previous findings, showing a clear concentration of high-priced homes near the coast and in densely populated areas.

### Correlations
We explore correlations between different attributes and the target variable (median house value) using a correlation matrix. This helps us identify potentially valuable features for our model.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729827553/scatter_matrix_plot.png)

Since we are having 11 numerical column, for visual manner we will get lot of plot(121). To avoid that we are focusing on few attributes that seem most correlated

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729827659/income_vs_house_value_scatterplot.png)

The correlation matrix reveals strong positive correlations between 'median_income' and 'median_house_value', suggesting its potential importance in predicting housing prices. Other attributes like 'total_rooms' and 'housing_median_age' also exhibit some correlation with the target variable.

#### Feature Engineering and Experimenting their combination
To improve model performance, we experiment with feature engineering by creating new features based on existing ones. We analyze the correlation of these new features with the target variable to assess their usefulness.

```js
housing['rooms_per_house'] = housing['total_rooms'] / housing['households']
housing['bedroom_ratio'] = housing['total_bedrooms'] / housing['total_rooms']
housing['people_per_house'] = housing['population'] / housing['households']
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729914845/corr_image.png)

The new features, like 'rooms_per_house' and 'bedroom_ratio', provide additional information about the housing characteristics and contribute to a better understanding of the dataset.

### Preparing Data for Machine Learning
Before training our model, we need to prepare the data by addressing missing values, handling categorical attributes, and scaling numerical features.

#### Data Cleaning  
We address missing values in the 'total_bedrooms' attribute by replacing them with the median value. However, to handle potential missing values in other columns with new datasets, we implement the **Imputer** function.
```js
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
// Finding the numerical in the Data Frame
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
// Tranforming to Data Frame
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns,index = housing_num.index)
```

#### Handling Text and Categorical columns
We handle the categorical attribute **'ocean_proximity'** by converting it into numerical values using both **Ordinal Encoder** and **OneHot Encoder**. These techniques enable the model to process categorical data effectively.

```js
// OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

// OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

#### Feature Scaling and Transformation
We explore different feature scaling techniques, namely **MinMaxScaler** and **StandardScaler**, to standardize the numerical attributes and improve model performance.

```js
// MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
min_max_scalar = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scalar.fit_transform(housing_num)
housing_num_min_max_scaled

// StandardScaler
from sklearn.preprocessing import StandardScaler
std_scalar = StandardScaler()
housing_num_std_scaled = std_scalar.fit_transform(housing_num)
housing_num_std_scaled
```

### Transformation Pipelines
To streamline the data preprocessing steps and ensure consistent execution order, we utilize **Pipeline** and **ColumnTransformer.**
```js
from sklearn.compose import ColumnTransformer
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms","total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]
cat_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'),OneHotEncoder(handle_unknown='ignore'))
preprocessing = ColumnTransformer([('num', num_pipeline ,num_attribs),
                                   ('cat',cat_pipeline, cat_attribs)])
```

We demonstrate the use of **make_pipeline()** and **column_transformer()** for simplified pipeline creation and highlight the advantages of using these techniques for complex data transformations.
We then combine all the preprocessing steps into a single **ColumnTransformer**, efficiently handling both numerical and categorical attributes

```js
def column_ratio(X):
    return X[:,[0]]/X[:,[1]]

def ratio_name(function_transformer, feature_names_in):
    return['ratio'] // Gives feature names

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy = 'median'),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy = 'median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler())

cluster_simil = ClusterSimilarity(n_clusters = 10, gamma = 1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population","households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],remainder = default_num_pipeline )// for remaining column - housing_median_age

housing_prepared = preprocessing.fit_transform(housing)
```

### Select and Train a Model

We start by training a Linear Regression model and evaluating its performance using RMSE. The model shows signs of underfitting, prompting us to explore more powerful models.

```js
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2) // -2 rounded to the nearest hundred
lin_rmse = mean_squared_error(housing_labels, housing_predictions,squared = False)
lin_rmse
// Output --> 68687.89176590036
```

We then train a Decision Tree Regressor, which overfits the training data, indicating the need for model validation techniques.
```js
from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions,squared = False)
tree_rmse
// Output --> 0.0
```

#### Evaluation using Cross - Validation
To address overfitting and obtain a more reliable performance estimate, we utilize cross-validation with 10 folds. This technique provides a robust evaluation of the Decision Tree Regressor, revealing its limitations in generalizing to unseen data.
```js
from sklearn.model_selection import cross_val_score
tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                             scoring = 'neg_root_mean_squared_error', cv = 10)
```

Next, we explore a Random Forest Regressor, which exhibits significantly better performance compared to the previous models.
```js
from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                               scoring = 'neg_root_mean_squared_error', cv = 10)
```

#### Model Fine-Tune
To further enhance the Random Forest Regressor's performance, we perform model fine-tuning using Grid Search and Randomized Search. These techniques explore different hyperparameter combinations to identify the optimal model configuration.
```js
// GridSearchCV
from sklearn.model_selection import GridSearchCV
full_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('random_forest', RandomForestRegressor(random_state=42))])

param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
    'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
    'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv = 3,
                           scoring = 'neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
cv_res.head()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729910963/Gridsearchcvtable.png)

```js
//RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}
rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)
rnd_search.fit(housing, housing_labels)

cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by = 'mean_test_score', ascending=False, inplace = True)
cv_res = cv_res[['param_preprocessing__geo__n_clusters',
                  'param_random_forest__max_features', 'split0_test_score',
                  'split1_test_score','split2_test_score','mean_test_score']]
cv_res.columns = ['n_clusters','max_features']+score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
cv_res.head()
```
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729911188/Randomized_search.png)

We analyze the results of these searches, highlighting the best hyperparameters and their impact on model performance.

#### Analyzing best models and their errors
We examine the feature importances of the best model, providing insights into the most influential features in predicting housing prices. This helps us understand the model's decision-making process and identify key factors driving price variations.

```js
final_model = rnd_search.best_estimator_
feature_importances = final_model['random_forest'].feature_importances_
feature_importances.round(2)
sorted(zip(feature_importances,
           final_model['preprocessing'].get_feature_names_out()),
           reverse = True)
```

#### Evaluating the Test Set

Finally, we evaluate the final model on the test set, using RMSE and confidence intervals to assess its generalizability and prediction accuracy.

```js
X_test = strat_test_set.drop('median_house_value', axis = 1)
y_test = strat_test_set['median_house_value'].copy()
final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared = False)
print(final_rmse)
// Output --> 41424.4

// We can compute a 95% confidence interval for the test RMSE
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc = squared_errors.mean(),
                        scale = stats.sem(squared_errors)))
//Output -->39275.4, 43467.3

// Showing how to compute a confidence interval for the RMSE
m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1+confidence)/2, df = m -1)
tmargin = tscore*squared_errors.std(ddof=1)/np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean+tmargin)
// Output --> 39275.4, 43467.3

//Alternatively, we can use Z-score rather than t-score. Since the test is too small, it won't make a huge difference.
zscore = stats.norm.ppf((1+confidence)/2)
zmargin = zscore * squared_errors.std(ddof = 1)/np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean+zmargin)
// Output --> 39276.1, 43466.7
```

### Launch, Monitor and Maintain our System
We discuss deployment strategies, highlighting the importance of monitoring the model's live performance, ensuring data quality, and maintaining backups for rollback capabilities.

```js
import joblib
joblib.dump(final_model, 'my_california_housing_model.pkl')
```
We showcase how to load the saved model and use it to predict housing prices for new data, demonstrating the model's practical applicability.

```js
final_model_reloaded = joblib.load("my_california_housing_model.pkl")
new_data = housing.iloc[:5]  // pretend these are new districts
predictions = final_model_reloaded.predict(new_data)
```

**new_data**
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729912426/new_data.png)

**Original value** of housing_labels.iloc[:5]
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729912643/original_value_housing_label.png)

**Predicted values**
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729912752/predicted.png)

By following these steps, we successfully built a machine learning model to predict housing prices in California. The project demonstrates the importance of data exploration, preprocessing, feature engineering, model selection, evaluation, and deployment considerations for creating robust and reliable machine learning solutions.
