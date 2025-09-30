from loading_dataframe import obtain_local_data
import pandas as pd

"""File path and initial data load"""
file_path = "failure_data.csv"
failure_df = obtain_local_data(file_path)

class DataFrameInfo:
    """A class for inspecting the properties of a pandas DataFrame."""   
    def __init__(self, df):
        """
        Initializes the DataFrameInfo class.

        Args:
            df (pd.DataFrame): The pandas DataFrame to inspect.
        """
        self.df = df

    def total_missing(self):
        """Returns the total number of missing values for each column."""
        return self.df.isnull().sum()

    def column_types(self):
        """Returns the data types of the DataFrame's columns."""
        return self.df.dtypes

    def df_info(self):
        """Displays information about the DataFrame's structure."""
        self.df.info()

    def df_description(self):
        """Returns descriptive statistics for numeric columns in dataFrame, ignoring nulls."""
        return self.df.describe()

    def obtain_shape(self):
        """Returns tuple: The rows and columns for the DataFrame."""
        shape = self.df.shape
        print(f"This dataset has {shape[0]} rows and {shape[1]} columns")
        return shape

    def distinct_counts(self):
        """Returns counts of unique values for categorical columns."""
        return self.df.select_dtypes(include=['category', 'object']).nunique()
   
    def extract_statistics(self):
        """Returns median, mean, and standard deviation for numeric columns."""
        return pd.DataFrame({
            'median': self.df.median(numeric_only=True),
            'std': self.df.std(numeric_only=True),
            'mean': self.df.mean(numeric_only=True)
        })

    def percentage_missing(self):
        """Returns the percentage of null values in each column."""
        return (self.df.isnull().sum() / len(self.df)) * 100
    
    def data_frame_full_info(self):
        """Calls other methods for abstraction purposes providing a full summary of dataframe."""
        print("\nDataFrame Info")
        self.df_info()
        
        print("\nShape")
        self.obtain_shape()
        
        print("\nColumn Data Types")
        print(self.column_types())
        
        print("\nTotal Missing Values")
        print(self.total_missing())
        
        print("\nPercentage of Missing Values")
        print(self.percentage_missing())
        
        print("\nDistinct Counts in Categorical Columns")
        print(self.distinct_counts())
        
        print("\nDescriptive Statistics")
        print(self.df_description())
        
        print("\nMedian, Mean, and Standard Deviation")
        print(self.extract_statistics())

