---
date: 2024-03-05 12:26:40
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

This Tableau project is a step-by-step learning experience in building dashboard projects using Tableau from requirements to professional dashboard like I do in my real-world projects.

Link for [Tableau - HR Dashboard](https://public.tableau.com/views/HR_Analytics_17259358090400/HRSummary?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

For downloading Data Generation code and Tableau File please visit [HR Analytics - Tableau(Github)](https://github.com/VijaikumarSVK/HR-Analytics---Tableau-Dashboard)

## User Story - HR Dashboard
As an HR manager, I want a comprehensive dashboard to analyze human resources data, providing both summary views for high-level insights and detailed employee records for in-depth analysis

### Summary View
The summary view should be divided into three main sections: Overview, Demographics, and Income Analysis

#### Overview
The Overview section should provide a snapshot of the overall HR metrics, including:<br>

1) Display the total number of hired employees, active employees, and terminated employees.<br>
2) Visualize the total number of hired and terminated employees over the years.<br>
3) Present a breakdown of total employees by department and job titles.<br>
4) Compare total employees between headquarters (HQ) and branches (New York is the HQ)<br>
5) Show the distribution of employees by city and state.<br>

#### Demographics
The Demographics section should offer insights into the composition of the workforce, including:<br>

1) Present the gender ratio in the company.:<br>
2) Visualize the distribution of employees across age groups and education levels.:<br>
3) Show the total number of employees within each age group.:<br>
4) Show the total number of employees within each education level.:<br>
5) Present the correlation between employees’s educational backgrounds and their performance ratings.:<br>

#### Income
The income analysis section should focus on salary-related metrics, including::<br>

1) Compare salaries across different education levels for both genders to identify any discrepancies or patterns.:<br>
2) Present how the age correlate with the salary for employees in each department.:<br>

### Summary View
- Provide a comprehensive list of all employees with necessary information such as name, department, position, gender, age, education, and salary.
- Users should be able to filter the list based on any of the available columns.


### Data Generation(Chat-GPT Prompts)
Python script to generate a realistic dataset of 8950 records for human resources. The dataset should include the following attributes:<br>
1) Employee ID: A unique identifier.<br>
2) First Name: Randomly generated.<br>
3) Last Name: Randomly generated.<br>
4) Gender: Randomly chosen with a 46% probability for ‘Female’ and a 54% probability for ‘Male’.<br>
5) State and City: Randomly assigned from a predefined list of states and their cities.<br>
6) Hire Date: Randomly generated with custom probabilities for each year from 2015 to 2024.<br>
7) Department: Randomly chosen from a list of departments with specified probabilities.<br>
8) Job Title: Randomly selected based on the department, with specific probabilities for each job title within the department.<br>
9) Education Level: Determined based on the job title, chosen from a predefined mapping of job titles to education levels.<br>
10) Performance Rating: Randomly selected from ‘Excellent’, ‘Good’, ‘Satisfactory’, ‘Needs Improvement’ with specified probabilities.<br>
11) Overtime: Randomly chosen with a 30% probability for ‘Yes’ and a 70% probability for ‘No’.<br>
12) Salary: Generated based on the department and job title, within specific ranges.<br>
13) Birth Date: Generated based on age group distribution and job title requirements, ensuring consistency with the hire date.<br>
14) Termination Date: Assigned to a subset of employees (11.2% of the total) with specific probabilities for each year from 2015 to 2024, ensuring the termination date is at least 6 months after the hire date.<br>
15) Adjusted Salary: Calculated based on gender, education level, and age, applying specific multipliers and increments.<br>
16) Be sure to structure the code cleanly, using functions where appropriate, and include comments to explain each step of the process.<br>  
