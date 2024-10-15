---
date: 2024-09-10 12:26:40
layout: post
title: Earthquake Data Analysis
subtitle: 'Unveiling Seismic Secrets: An In-Depth Analysis of Earthquake Data'
description: Harnessing Python and Machine Learning to Explore Earthquake Patterns and Predict Magnitude
image: https://res.cloudinary.com/dqqjik4em/image/upload/v1728971751/Earthquake_cover_image_2.jpg
optimized_image: https://res.cloudinary.com/dqqjik4em/image/upload/f_auto,q_auto/Earthquake_cover_image_2
category: Data Science
tags:
  - ML
  - EDA
author: Vijai Kumar
vj_layout: false
vj_side_layout: true
---

Earthquakes, a testament to the immense power hidden beneath our feet, have captivated and terrified humanity for millennia. These seismic events, though destructive, hold a fascinating complexity that begs to be understood. This project delves into the depths of historical earthquake data, leveraging the flexibility of Python and the power of machine learning to uncover hidden patterns and explore the potential for predicting earthquake magnitude.

## Data and Methodology
Our journey begins with a rich dataset sourced from [mention data source], documenting a comprehensive record of earthquakes around the world. Each entry in the dataset captures key characteristics of an earthquake, including:

1. **Date and Time:** Pinpointing when the earthquake struck.

2. **Region:** Providing a general location of the seismic event.

3. **Magnitude:** Quantifying the earthquake's energy release on the Richter scale.

4. **Depth** Indicating the depth below the Earth’s surface where the earthquake originated.

5. **Latitude and Longitude:** Precisely mapping the earthquake's epicenter.

Before embarking on our analysis, the raw data underwent a refinement process:

- **Date Transformation:** To facilitate temporal analysis, the 'Date' column, initially stored as text, was converted into a datetime object using pd.to_datetime().

- **Numerical Extraction:** Latitude and Longitude information, embedded within strings, were extracted and converted to numerical values, preparing them for further analysis.

```js
// Converting Date column into pandas date type
EQ_df['Date'] = pd.to_datetime(EQ_df['Date'])
print('Data frame min date is', min(EQ_df['Date']),'and max date is',max(EQ_df['Date']))

// Extracting Latitude and Longitude from the data
EQ_df['Latitude'] =  EQ_df['Latitude'].str.extract(r'(\d+\.\d+)')[0].astype(float) * EQ_df['Latitude'].str.extract(r'([NS])')[0].map({'N': 1, 'S': -1})
EQ_df['Longitude'] =  EQ_df['Longitude'].str.extract(r'(\d+\.\d+)')[0].astype(float) * EQ_df['Longitude'].str.extract(r'([EW])')[0].map({'E': 1, 'W': -1})

```




Vivamus sagittis lacus vel augue rutrum faucibus dolor auctor. Duis mollis, est non commodo luctus, nisi erat porttitor ligula, eget lacinia odio sem nec elit. Morbi leo risus, porta ac consectetur ac, vestibulum at eros.

## Code

Cum sociis natoque penatibus et magnis dis `code element` montes, nascetur ridiculus mus.

```js
// Example can be run directly in your JavaScript console

// Create a function that takes two arguments and returns the sum of those arguments
var adder = new Function("a", "b", "return a + b");

// Call the function
adder(2, 6);
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

Nullam id dolor id nibh ultricies vehicula ut id elit. Sed posuere consectetur est at lobortis. Nullam quis risus eget urna mollis ornare vel eu leo.
