# zillow_clustering_project
# Zillow Project


## Project Planning
- Use information from the zillow data base to acquire, prepare, explore, and make models to predict the value of single family residential properties. 


## Project Goals
- Create a jupyter notebook that does the following:
  - Acquires and prepares the data
  - Explores the data
  - Creates and refines a model
  - Offers conclusions and a way forward
- Create python modules that can assist in the process of making the jupyter notebook and make it repeatable
   - wrangle.py
   - explore.py 
   - scaling.py
- Ask questions during the exploration phase to better understand what could be factoring into the target variable of tax value dollar count
  - answer those questions with statistical tests and visualizations
- Construct machine learning models for predicting the tax value dollar count of the properties
  - run most effective machine learning model against test
- ensure everything is documented and annotated
- Give 5 mminute presentation to Zillow Data Science Team


## Business Goals
- Find key drivers for property value for single family residential properties
- Deliver a report to the Zillow Data Science Team with adequate documentation and comments
- Construct a machine learning model that can be used to predict the property values for single family residential properties

## Audience
- Zillow Data Science Team

## Deliverables
- jupyter notebook containing the final report
- python modules that can be used to reproduce the work 
- scratch notebooks that can be refered to for my work
- live presentation of final notebook
- readme file explaining project



## Data dictionary
- factors that were brought from the zillow data set 
<img src="images/data_dictionary.png" width = 750>


# Project plan
- Acquire the data from the codeup db. Create an acquire.py file
- Clean and prepare the data for the exploration phase. Create a prepare.py file to recreate the work
- Explore the data and ask questions to clarify what is actually happening. 
  - ensure to properly annotate, comment, and use markdowns
  - write out each null and alternative hypothesis
  - visualize the data
  - run statistical test on the data
- create at least 3 different machine learning models
- choose the model that performs the best
  - evaluate on test
- Deliver final presentation to Zillow Data Science Team


## Executive Summary
- The factors that most drive tax value dollar count (from most to least important) are total square feet, bedroom count, age, and location
- Currently, our best shot at predicting the tax value dollar count of a property is my polynomial regression model.

## Reproduce the project
- In order to reproduce the project, you will need
  - env.py file that will enable you access to the CodeUp database
  - credentials for codeup database
  - All other files contained in this repository
  
