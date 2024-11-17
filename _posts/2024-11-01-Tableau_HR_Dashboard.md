---
date: 2024-11-01 12:26:40
layout: post
title: Tableau - HR Dashboard
<!-- subtitle: This Tableau project is a step-by-step learning experience in building dashboard projects using Tableau from requirements to professional dashboard like I do in my real-world projects. -->
description: This Tableau project provides a step-by-step guide to building professional dashboards, mirroring real-world scenarios.
image: https://res.cloudinary.com/dqqjik4em/image/upload/v1727449780/HR_dashboard.png
optimized_image: https://res.cloudinary.com/dqqjik4em/image/upload/f_auto,q_auto/HR_dashboard
category: Visualization
tags:
  - Tableau
  - Chat GPT
author: Vijai Kumar
paginate: true
vj_layout: false
vj_side_layout: true
---

Link for [Tableau - HR Dashboard](https://public.tableau.com/views/HR_Analytics_17259358090400/HRSummary?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

For downloading Data Generation code and Tableau File please visit [HR Analytics - Tableau(Github)](https://github.com/VijaikumarSVK/HR-Analytics---Tableau-Dashboard)

## User Story - HR Dashboard
As an HR manager, my objective is to develop a comprehensive dashboard capable of providing both high-level and granular insights into our human resources data. This dashboard will facilitate strategic decision-making by offering summary views for overall workforce trends, while also allowing for deep dives into individual employee records when more detailed analysis is required.

### Summary View
The summary view of the dashboard will be structured into three primary sections: Overview, Demographics, and Income Analysis. Each section will be designed to provide key insights at a glance.

#### Overview
This section aims to provide a comprehensive overview of the company's key HR metrics.  I have designed it to offer insights into the workforce composition and distribution at a glance. The section encompasses the following aspects:

1. **Employee Headcount:**  A clear presentation of the total number of hired, active, and terminated employees, providing a snapshot of the current workforce size and turnover.

2. **Hiring and Termination Trends:**  A visualization depicting the total number of hired and terminated employees over the years, enabling the identification of historical trends in workforce growth and attrition.

3. **Departmental and Job Title Analysis:** A breakdown of the total employee count by department and respective job titles. This analysis facilitates understanding workforce distribution across different departments and roles.

4. **Headquarters vs. Branch Comparison:** A comparison of the total number of employees based in the headquarters (HQ) versus those situated in branches, with New York designated as the HQ location. This comparison highlights the concentration of the workforce.

5. **Geographical Distribution:** A comprehensive view of employee distribution by city and state, providing insights into the geographical reach and concentration of the workforce.


#### Demographics
This section delves into the composition of the workforce, providing insights into:
1. **Gender Distribution:**  An analysis of the gender ratio within the company.
2. **Age and Education Demographics:** Visual representations of employee distribution across various age groups and education levels.
3. **Age Group Breakdown:**  A numerical representation of the total number of employees within each designated age group.
4. **Education Level Breakdown:** A numerical representation of the total number of employees within each designated education level.
5. **Performance by Education:** An examination of the correlation between employee educational backgrounds and their corresponding performance ratings.

#### Income
This section centers on salary-related metrics, with a focus on:

 1. **Salary by Education and Gender:** A comparative analysis of salaries across different education levels, disaggregated by gender, to ascertain any potential discrepancies or patterns.
 2. **Salary by Age and Department:** An examination of the correlation between age and salary for employees within each department.

### Employee Records View
This section will house a comprehensive roster encompassing all employees, complete with pertinent details including:

 - **Employee Information:**  Name, Department, Position, Gender, Age, Education, and Salary.

To facilitate efficient data exploration, users will have the ability to filter the roster based on any of the aforementioned columns.

![alt text](https://res.cloudinary.com/dqqjik4em/image/upload/v1727491255/HR_Details.png)

### Data Generation(Chat-GPT Prompts)
This Python script generates a synthetic dataset containing 8950 employee records for human resources purposes. The dataset incorporates a degree of realism by adhering to specified probabilities and correlations between attributes.

**Generated Attributes:**

1. **Employee ID:** A unique identifier assigned to each employee.
2. **First Name:** Randomly generated.
3. **Last Name:** Randomly generated.
4. **Gender:** Randomly assigned with a 46% probability for 'Female' and a 54% probability for 'Male', reflecting a plausible gender distribution.
5. **State and City:** Randomly selected from a predefined list of states and their corresponding cities, ensuring geographical diversity.
6. **Hire Date:** Randomly generated within a specified range (2015-2024), employing custom probabilities for each year to simulate hiring trends.
7. **Department:** Randomly assigned from a list of departments with predefined probabilities, reflecting the organizational structure.
8. **Job Title:** Randomly selected based on the assigned department, adhering to probabilities associated with each job title within that department.
9. **Education Level:** Determined based on the assigned job title, utilizing a predefined mapping between job titles and typical education level requirements.
10. **Performance Rating:**  Randomly allocated from a set of ratings ('Excellent', 'Good', 'Satisfactory', 'Needs Improvement') with specific probabilities, representing a performance distribution.
11. **Overtime:**  Randomly assigned with a 30% probability for 'Yes' and a 70% probability for 'No', simulating overtime patterns.
12. **Salary:** Generated within a range determined by the assigned department and job title, reflecting realistic salary structures.
13. **Birth Date:** Generated to align with age group distribution and job title requirements, ensuring consistency with the employee's hire date.
14. **Termination Date:**  Assigned to a subset of employees (11.2%) to simulate employee turnover. The termination year is randomly selected from 2015 to 2024 with specific probabilities, while ensuring a minimum six-month gap between the hire date and termination date.
15. **Adjusted Salary:** Calculated by factoring in gender, education level, and age, applying specific multipliers and increments to simulate potential pay disparities.
