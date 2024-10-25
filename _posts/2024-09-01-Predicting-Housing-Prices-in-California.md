---
date: 2024-09-01 12:26:40
layout: post
title: House Price Prediction in California
subtitle: An End-to-End Machine Learning Project
description: A machine learning model predicting California house prices, optimized through data preprocessing and tuning.
image: https://res.cloudinary.com/dqqjik4em/image/upload/v1729821798/House_price_predicition_cover_image.jpg
optimized_image: https://res.cloudinary.com/dqqjik4em/image/upload/f_auto,q_auto/House_price_predicition_cover_image
category: Data Science
tags:
  - ML
  - EDA
  - Predictive Analytics
author: Vijai Kumar
vj_layout: false
vj_side_layout: true
---

**Executive Summary:** This project utilizes machine learning techniques to predict housing prices in California. Using the California housing dataset, the project explores data preprocessing, feature engineering, model selection, and evaluation. The blog details the process, highlighting key insights and challenges encountered. Finally, deployment strategies and ongoing maintenance considerations are discussed.

### House Price Prediction (California)
#### Fetching Data
The first step in any machine learning project is gathering the data. For this project, we'll be using the California housing dataset, which contains information about various housing characteristics and their corresponding prices.

We load the data into a pandas DataFrame and explore its structure using **info()** and **describe()** methods. This allows us to understand the data types of each attribute, identify missing values, and gain insights into the data's distribution.

#### Data Visualization
Visualizing the data is crucial for understanding its underlying patterns. We create histograms to examine the distribution of each numerical attribute. This helps us identify potential issues like skewed distributions or capped values, which may impact model performance.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729822514/attribute_histogram_plots.png)

The histograms reveal interesting insights:
1.**Median income:** The data is capped, with the highest value at 15, representing $150,000. This might affect the model's ability to predict prices accurately for high-income areas.
2.	**Housing median value:** The housing prices are also capped, possibly limiting the model's prediction range.
3.	**Skewed Distributions:** Many histograms exhibit right-skewness, potentially challenging certain machine learning algorithms in detecting patterns.

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
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729828459/corr_image.png)

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


<!--
> Curabitur blandit tempus porttitor. Nullam quis risus eget urna mollis ornare vel eu leo. Nullam id dolor id nibh ultricies vehicula ut id elit.

Etiam porta **sem malesuada magna** mollis euismod. Cras mattis consectetur purus sit amet fermentum. Aenean lacinia bibendum nulla sed consectetur.

## Inline HTML elements

HTML defines a long list of available inline tags, a complete list of which can be found on the [Mozilla Developer Network](https://developer.mozilla.org/en-US/docs/Web/HTML/Element).

- **To bold text**, use `<strong>`.
- *To italicize text*, use `<em>`.
- Abbreviations, like <abbr title="HyperText Markup Langage">HTML</abbr> should use `<abbr>`, with an optional `title` attribute for the full phrase.
- Citations, like <cite>&mdash; Thomas A. Anderson</cite>, should use `<cite>`.
- <del>Deleted</del> text should use `<del>` and <ins>inserted</ins> text should use `<ins>`.
- Superscript <sup>text</sup> uses `<sup>` and subscript <sub>text</sub> uses `<sub>`.

Most of these elements are styled by browsers with few modifications on our part.

# Heading 1

## Heading 2

### Heading 3

#### Heading 4

Vivamus sagittis lacus vel augue rutrum faucibus dolor auctor. Duis mollis, est non commodo luctus, nisi erat porttitor ligula, eget lacinia odio sem nec elit. Morbi leo risus, porta ac consectetur ac, vestibulum at eros.

## Code

Cum sociis natoque penatibus et magnis dis `code element` montes, nascetur ridiculus mus.

```js
// Example can be run directly in your JavaScript console

// Create a function that takes two arguments and returns the sum of those arguments
var adder = new Function("a", "b", "return a + b");

// Call the function
adder(2, 6);
// > 8
```

Aenean lacinia bibendum nulla sed consectetur. Etiam porta sem malesuada magna mollis euismod. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa.

## Lists

Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aenean lacinia bibendum nulla sed consectetur. Etiam porta sem malesuada magna mollis euismod. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus.

* Praesent commodo cursus magna, vel scelerisque nisl consectetur et.
* Donec id elit non mi porta gravida at eget metus.
* Nulla vitae elit libero, a pharetra augue.

Donec ullamcorper nulla non metus auctor fringilla. Nulla vitae elit libero, a pharetra augue.

1. Vestibulum id ligula porta felis euismod semper.
2. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.
3. Maecenas sed diam eget risus varius blandit sit amet non magna.

Cras mattis consectetur purus sit amet fermentum. Sed posuere consectetur est at lobortis.

Integer posuere erat a ante venenatis dapibus posuere velit aliquet. Morbi leo risus, porta ac consectetur ac, vestibulum at eros. Nullam quis risus eget urna mollis ornare vel eu leo.

## Images

Quisque consequat sapien eget quam rhoncus, sit amet laoreet diam tempus. Aliquam aliquam metus erat, a pulvinar turpis suscipit at.

![placeholder](https://placehold.it/800x400 "Large example image")
![placeholder](https://placehold.it/400x200 "Medium example image")
![placeholder](https://placehold.it/200x200 "Small example image")

## Tables

Aenean lacinia bibendum nulla sed consectetur. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Upvotes</th>
      <th>Downvotes</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>Totals</td>
      <td>21</td>
      <td>23</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>Alice</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Charlie</td>
      <td>7</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

Nullam id dolor id nibh ultricies vehicula ut id elit. Sed posuere consectetur est at lobortis. Nullam quis risus eget urna mollis ornare vel eu leo. -->
