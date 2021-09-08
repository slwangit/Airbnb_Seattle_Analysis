# Airbnb Seattle Analysis

## Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation
In this project, there are no requirements for libraries beyond the Anaconda distribution of Python.
The code is recommended to be run using Python version 3.*.

The libraries used:<br>
* numpy<br>
* pandas<br>
* matplotlib.pyplot<br>
* seaborn<br>
* folium<br>
* statsmodels.tsa<br>
* nltk<br>
* sklearn

## Project Motivation<a name="motivation"></a>
For this project, I was interested in exploring Airbnb listings, reviews, and pricing in Seattle.  
Proposed questions:
1. What are the peak times of the year to visit Seattle?
2. How does price change across the year? Is there any trend or seasonality?
3. How customers feel about their stays in Seattle? Positive or negative? What are the keywords in their reviews?
4. Can we predict price based on the host and room data?
5. What factors of homestays contribute to salary?

For question1-3, I conducted a comprehensive exploratory data analysis(EDA) to give an overview of pricing, sentiment, and room distribution.
Visualization and descriptive analysis can be found in eda_utility.py file.
For question 4&5, I conducted a predictive modeling on the listing data using linear regression model. Details can be found in predictive_utility.py.

## File Descriptions<a name="files"></a>
There are two `.py` file and one notebook file available here to showcase work related to the above questions. The 2 utility `.py` files are functions deployed for EDA and predictive analysis.
For the whole workflow, results, and interpretation, please refer to the `airbnb_analysis.ipynb`.

## Results
The main findings and report can be found t the post available [here](link).

## Licensing, Authors, and Acknowledgements<a name="licensing"></a>
The datasets for this project was obtained from Kaggle. You can find licensing and other descriptive information at the source link available [here](https://www.kaggle.com/airbnb/seattle?select=calendar.csv).
Also, I must give credit to Stack Overflow for the coding part and Medium for the statistical part. 
All the code here are open to the public. Thus, feel free to use it as you would like.
