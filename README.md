# Manufacturing Process Analysis and Failure Prediction

## Table of Contents
1. [Objective](#objective)
2. [Project Description](#project-description)
3. [Installation Instructions](#installation-instructions)
4. [Usage Instructions](#usage-instructions)
5. [File Structure](#file-structure)
6. [Key Findings](#key-findings)
7. [License](#license)

## Objective
The aim of this project is to determine the optimal operating conditions for a manufacturing process through the analysis of a dataset, including information about machine operation settings, product quality, and various failure types. The manufacturing data within a dataframe is to be assessed through the utilisation of multiple python libraries including numpy, pandas, pyplot and seaborn. 

## Project Description
This project is focused on analyzing manufacturing process data stored in an AWS RDS database to identify patterns and risk factors associated with machine failures. The project involves the following processes:
- Data Extraction: Manufacturing data is retrieved into local storage from an AWS RDS database.
- Data Preprocessing: The data is cleaned and transformed to aid analysis, with methods utilised to handle missing values, reduce skewness, and remove outliers.
- EDA: The dataframe is summarised and assessments are made of the data distribution and any correlations present.
- Visualization Tools: Charts are created to display the patterns and key insights.
- Failure Insights: The root causes of the machine failures are investigated with relationships identified between operating conditions and different types of failures of the manufacturing process.

## Installation Instructions
To run this project locally, ensure you have run the following commands to install necessary dependencies: 
- *python -m venv env*
- *source env/bin/activate* (Use env\Scripts\activate on Windows)
- pip install -r requirements.txt
- pip install PyYAML

1. Clone the repository: git clone https://github.com/Fenlon37/exploratory-data-analysis---the-manufacturing-process414.git
2. Navigate to the project directory using the *cd* command.
3. Make sure python (3.6 or later) is installed using the *python --version* command.
4. Install required packages.
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
1. Run the final_df.py script using the following code *python final_df.py*
2. Use the DataFrameInfo class to generate summary statistics.
3. Visualize data and gain insights using methods in the Plotter class.
4. Handle missing values, skewed distributions, and outliers with the DataTransform class.
#### Failure Analysis
1. Compute operating ranges for key metrics like air temperature, torque, and tool wear.
2. Investigate failure causes and identify high-risk operating conditions.
#### Overview
- Please see the EDA notebook file for an overview of the project and the reasoning behind the coding decisions. - TBA

## File structure
    
    ├── db_utils.py                # Database data extraction

    ├── loading_dataframe.py       # DataFrame loaded
   
    ├── DataFrameInfo_df.py        # DataFrame basic information
    
    ├── DataTransform_df.py        # DataFrame transformations
    
    ├── Plotter_df.py              # Visualisations and statistical analysis
    
    ├── FailureAnalysis_df.py      # Analysis of tool failures    
    
    ├── final_df.py                # Activation of commands
    
    └── README.md                  # README file


## Key Findings
- 3.5% of manufacturing processes experienced failures.
- Head dissipation failure (HDF) was the most common cause of failure.
- Higher tool wear was strongly associated with machine failures.
- On average machine failures also occurred more frequently when torque was higher and rotational speed was lower, but overstrain failure (OSF), had the opposite relationship.
- Operating ranges were broadly comparable regardless of the product quality.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
