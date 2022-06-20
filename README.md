
# Zillow Clustering Project


## Project Planning
- Use information from the zillow data base to acquire, prepare, explore, and make models to predict the logerror of single family residential properties. 


## Project Goals
### Answer the question : What are some of the reasons for log error and how can we reduce it?
- Ask questions during the exploration phase to better understand what could be factoring into the target variable of logerror
  - answer those questions with statistical tests and visualizations
- These answers will be used to hopefully reduce log error in the future
- Construct machine learning models for predicting the logerror of the properties
  - run most effective machine learning model against test
- ensure everything is documented and annotated
- Give 5 mminute presentation to Zillow Data Science Team


## Business Goals
- Find key drivers for logerror for single family residential properties
- Deliver a report to the Zillow Data Science Team with adequate documentation and comments
- Construct a machine learning model that can be used to predict the logerror for single family residential properties
- Try to beat the current machine learning model to reduce log error in the future
- This will give more accurate property values to consumers when they are attempting to buy a home

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

## Initial Hypotheses:
- Logerror has a relationship with the location of the property
- Logerror is strongly related to the time of year that the property was sold.
- Log error is related to the value of the property.
- Log error and the age of the property are related.

## Executive Summary
- A few key factors have a relationship with log error, such as:
  - The location of the property
  - When the property was purchased
  - The age of the property
- The value of a property doesn't seem to be a key factor of log error
- It seems that the more properties that are sold during a time frame the lower the log error.


## Reproduce the project
- In order to reproduce the project, you will need
  - env.py file that will enable you access to the CodeUp database
  - credentials for codeup database
  - All other files contained in this repository
  - clone this repository to your local  machine
  - Understanding of what log error is (can be found in data dictionary)
  
