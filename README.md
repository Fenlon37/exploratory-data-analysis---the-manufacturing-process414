# Project Title: AI Core Manufacturing Data Analysis

## Table of Contents
1. [Project Description](#project-description)
2. [Installation Instructions](#installation-instructions)
3. [Usage Instructions](#usage-instructions)
4. [File Structure](#file-structure)
5. [Key Findings](#key-findings)
6. [License](#license)

---

## Project Description
This project is focused on analyzing manufacturing process data stored in an AWS RDS database and involves the following processes:
- Data Extraction: Manufacturing data is retrieved into local storage from an AWS RDS database.
- Data Preprocessing: The data is cleaned and transformed to aid analysis.
- EDA: The dataframe is summarised and assessments are made of the data distribution and any correlations present.
- Visualization Tools: Charts are created to display the patterns and key insights.
- Failure Analysis: The root causes of the machine failures are investigated with potential risk factors identified.

**Objective**:
The aim of this project was to analyse manufacturing data within a dataframe through the utilisation of multiple python libraries including numpy, pandas, pyplot and seaborn. Different classes were created to perform tasks including the extraction, processing, analysis and visualisation of the data, to enable management to ascertain the optimal operating conditions for the manufacturing process.

**What I Learned**:

---

## Installation Instructions
To run this project locally, ensure you have the following dependencies installed:

1. Clone the repository: git clone https://github.com/Fenlon37/exploratory-data-analysis---the-manufacturing-process414.git
2. Navigate to the project directory using the *cd* command.
3. Make sure python (3.6 or later) is installed using the *python --version* command.
4. Install required packages...
5. Create a credentials.yaml file with the following format:
RDS_HOST: 
RDS_PASSWORD: 
RDS_USER: 
RDS_DATABASE: 
RDS_PORT: 

## Usage instructions
#### Data Extraction and Transformation
1. Run the db_utils.py script to extract data from the RDS database: Within bash type the following code *python db_utils.py*
2. Save the extracted data to a local CSV for faster future analysis.
3. Run the loading_dataframe.py script using the following code *python loading_dataframe.py*
4. Use the DataTransform class to clean and preprocess the dataset.
#### Exploratory Data Analysis (EDA)
1. Run the eda_final_failure3_df.py script Using the following code using the following code *python eda_final_failure3_df.py*
2. Use the DataFrameInfo class to generate summary statistics.
3. Visualize data using methods in the Plotter class.
4. Handle missing values, skewed distributions, and outliers with DataFrameTransform.
#### Insights and Analysis
1. Compute operating ranges for key metrics like air temperature, torque, and tool wear.
2. Investigate failure causes and identify high-risk operating conditions.

## File structure

## Key Findings

## License
