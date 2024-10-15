---
date: 2024-09-10 12:26:40
layout: post
title: Earthquake Analysis
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

Earthquakes is a testament to the immense power hidden beneath our feet, have captivated and terrified humanity for millennia. These seismic events, though destructive, hold a fascinating complexity that begs to be understood. This project delves into the depths of historical earthquake data, leveraging the flexibility of Python and the power of machine learning to uncover hidden patterns and explore the potential for predicting earthquake magnitude.

The complete Python code for this analysis is available on my <b><a href="https://github.com/VijaikumarSVK/Earthquake-Analysis">GitHub</a></b>

## Data and Methodology
Our journey begins with a rich dataset sourced from <b><a href="https://seismic.pmd.gov.pk/">Pakistan Meteorological Department</a></b>, documenting a comprehensive record of earthquakes around the world. Each entry in the dataset captures key characteristics of an earthquake, including:

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
## Exploratory Data Analysis

Armed with a clean and structured dataset, we embarked on Exploratory Data Analysis (EDA), a crucial step in any data science endeavor. EDA allows us to unravel the story hidden within the numbers, revealing intriguing patterns and relationships.

#### Temporal Analysis: Earthquakes Through Time
Visualizing earthquake occurrences over time unveils the dynamic nature of our planet. By plotting earthquake frequency across years, months, and even hours, we gain valuable insights into temporal trends and potential seasonality.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729006420/Year_wise_EQ.jpg)
![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729006645/Month_wise_EQ.jpg)

#### Regional Insights: Where do Earthquakes occurs more?

Investigating earthquake magnitudes by region highlights the uneven distribution of seismic activity across the globe. By calculating average magnitudes for each region, we can identify areas prone to more powerful earthquakes and those relatively calmer.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729007252/Country_wise_EQ.jpg)

#### Depth vs. Magnitude
Delving deeper, we explored the relationship between an earthquake’s depth and its magnitude. A scatter plot, visualizing these two variables, provides insights into whether deeper earthquakes tend to be more or less intense.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729008035/Dp_Vs_mag_wise_EQ.jpg)

#### Detection Methods
Analyzing the distribution of earthquake detection methods — automatic vs. manual — offers a glimpse into how these events are captured and verified. Understanding the prevalence of each method sheds light on potential biases or inconsistencies in the data.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1729008228/Detection_method.jpg)


## Feature Engineering
Feature engineering is for transforming raw data into a format suitable for machine learning algorithms. This crucial step often involves creating new features from existing ones to enhance the model's predictive power. In our analysis, we engineered two new features:

- **Hour of the Day:** Recognizing that earthquake occurrences might exhibit diurnal patterns, we extracted the hour from the time column. This feature could reveal if earthquakes are more likely to strike at certain times.

- **Country:** Using a Geocoding API, we mapped the latitude and longitude of each earthquake to its corresponding country. This added geographical context could unveil regional variations in earthquake frequency and magnitude.

```js
# Using Geocode API for find the country using Region
row = 1
def location_finder(lat,lon):
    global row
    row = row + 1
    print(row)
    try:        
        response = requests.get(f'https://geocode.maps.co/reverse?lat={str(lat)}&lon={str(lon)}&api_key = API_KEY')
        res = response.json()
        try:
            return res['address']['country']
        except:
            return 'No country'
        except:
            return 'No country'

EQ_df['Country'] = np.vectorize(location_finder)(EQ_df['Latitude'],EQ_df['Longitude'])
```

## Predictive Modeling

With our data fortified with insightful features, we set out to build a machine learning model capable of predicting earthquake magnitude. For this task, we chose the Random Forest Regressor, a robust algorithm well-suited for handling complex, non-linear relationships in data.

The model training involved splitting the dataset into training and testing sets. The Random Forest was trained on the training set and its performance evaluated on the unseen testing set using:

- **Mean Squared Error (MSE):** Quantifies the average squared difference between predicted and actual magnitudes. Lower MSE values indicate better model accuracy. <br>
<b>MSE Score: 15.68%</b>

- **R-squared (R2) Score:** Using a Geocoding API, we mapped the latitude and longitude of each earthquake to its corresponding country. This added geographical context could unveil regional variations in earthquake frequency and magnitude.<br>
<b>R2 Score: 92.17%</b>

## Conclusion and Looking Ahead
This analysis has provided valuable insights into earthquake patterns and the potential for predicting their magnitude. We've uncovered temporal trends, regional variations, and relationships between depth and magnitude, showcasing the power of data exploration. Our predictive model, while a promising start, can be further enhanced through.


1. **Expanded Datasets:** Incorporating a larger dataset with a wider temporal and geographical scope can improve model accuracy and generalizability.

2. **Additional Features:** Adding features like fault line proximity, soil types, and tectonic plate movements can provide valuable geological context for prediction.

3. **Algorithm Exploration:** Exploring alternative machine learning algorithms, such as gradient boosting or neural networks, might unlock even greater predictive power.

The ability to forecast earthquakes holds immense potential for disaster preparedness, infrastructure planning, and ultimately, saving lives. This exploration serves as a stepping stone towards a future where we can better anticipate and mitigate the impacts of these powerful natural events.
