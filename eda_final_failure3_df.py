from statsmodels.graphics.gofplots import qqplot
from loading_dataframe import obtain_local_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""File path and initial data load"""
file_path = "failure_data.csv"
failure_df = obtain_local_data(file_path)

class DataFrameInfo:
    """A class for inspecting properties and characteristics of a pandas DataFrame."""   
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

class DataTransform:
    """
    DataTransform class for data cleaning and transformation of the pandas dataframe
    
    Attributes:
        df (pd.DataFrame): The pandas DataFrame to transform.
        snapshots (dict): A dictionary to save copies of the DataFrame for reference.
    """    
    def __init__(self, df):
        """
        Initializes the DataTransform class.

        Args:
            df (pd.DataFrame): The pandas DataFrame to transform.
        """       
        self.df = df
        self.snapshots = {}

    def delete_irrelevant_columns(self, columns):
        """
        Deleting columns not useful within analysis from dataframe.
        Args:
            columns (list): List of column names to delete.       
        """
        self.df.drop(columns=columns, inplace=True)

    def save_copy(self, label):
        """
        Saves a copy of the current DataFrame with a label for comparison.
        Args:
            label (str): A label for identifying the saved DataFrame.
        """   
        self.snapshots[label] = self.df.copy()
        print(f"Snapshot '{label}' saved.")

    def replace_blank_with_nan(self):
        """Replaces the empty cells of dataframe with NaN"""
        self.df.replace({" ": np.nan, "": np.nan}, regex=True, inplace=True)
        print("Null values after replacing blanks with NaN:")
        print(self.df.isnull().sum())
    
    def fill_missing_with_median(self, columns):
        """
        Replaces NaN cells with the median of respective column
        Args:
            columns (list): List of column names to fill missing values.        
        """
        for column in columns:
            if column in self.df.columns:
                median_value = self.df[column].median()
                self.df[column].fillna(median_value, inplace=True)

    def remove_missing(self, subset):
        """
        Drops remaining rows still containing NaN cells in specified column subset.
        Args:
            subset (list): List of column names to check for NaN values.                
        """ 
        before_rows = len(self.df)
        self.df.dropna(subset=subset, inplace=True)
        after_rows = len(self.df)
        print(f"Dropped {before_rows - after_rows} rows due to missing values.")  

    def transform_type_column_values(self):
        """Transforming "Type" column values in the DataFrame."""
        if "Type" in self.df.columns:
            self.df["Type"].replace({"H": "High", "M": "Medium", "L": "Low"}, inplace=True)

    def transform_failure_column_values(self):
        """Transforming columns 9-14 that are of Boolean format to "Non-failure" and "Failure"."""
        failure_columns = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        self.df[failure_columns].replace({0: "Non-failure", 1: "Failure"}, inplace=True)

    def convert_columns_to_categorical(self):
        """Converting failure columns 9-14 to categorical data type"""
        failure_columns = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]        
        for column in failure_columns:
           if column in self.df.columns:
             self.df[column] = self.df[column].astype('category')

    def transform_skewed_columns(self, threshold=1):
        """
        Applies transformations to skewed numeric columns.
        
        Args:
            threshold (float): Skewness threshold for applying transformation.
        """
        numeric_columns = self.get_numeric_columns()
        for column in numeric_columns:            
            skewness = self.df[column].skew()
            print(f"Skewness of {column}: {skewness}")
            if abs(skewness) > threshold:
                self.df[column] = np.log1p(self.df[column])

    def handle_outliers(self, z_thresh=3):
        """
        Removes rows with outliers based on Z-scores.
        Args:
            z_thresh (float): Z-score threshold for identifying outliers.               
        """
        numeric_columns = self.get_numeric_columns()      
        for column in numeric_columns:
            z_scores = (self.df[column] - self.df[column].mean()) / self.df[column].std()
            self.df = self.df[z_scores.abs() < z_thresh]

    def remove_highly_correlated(self, threshold=0.9):
        """
        Removes highly correlated columns.
        
        Args:
            threshold (float): Correlation threshold for removal.
        """
        numeric_columns = self.get_numeric_columns()
        numeric_df = self.df[numeric_columns]  
        corr_matrix = numeric_df.corr()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.df.drop(columns=to_drop, inplace=True)

    def get_numeric_columns(self):
        """Returns list of numeric columns explicitly excluding categorical and deleted."""
        exclude_columns = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_columns if col not in exclude_columns]    

class Plotter:
    """Plotter class for visualizing data distributions and relationships within a pandas DataFrame."""
    def __init__(self, df):
        """
        Initializes the Visualization class.

        Args:
            df (pd.DataFrame): The pandas DataFrame to visualize.
        """
        self.df = df

    def visualize_nulls(self):
        """Plots null value counts for each column."""
        nulls = self.df.isnull().sum()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=nulls.index, y=nulls.values)
        plt.title("Null Value Counts")
        plt.xticks(rotation=45)
        plt.show()

    def visualize_skew(self):
        """Plots the skewness of numeric columns."""
        numeric_columns = self.get_numeric_columns()
        numeric_df = self.df[numeric_columns]
        skewness = numeric_df.skew() 
        skewness.plot(kind="bar", title="Skewness of Numeric Columns", figsize=(10, 6))
        plt.show()

    def visualize_outliers(self, column):
        """Generates a boxplot for a given column to visualize outliers."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=self.df[column])
        plt.title(f"Outliers in {column}")
        plt.show()
    
    def get_numeric_columns(self):
        """Returns numeric columns explicitly excluding categorical and deleted."""
        exclude_columns = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_columns if col not in exclude_columns]

    def visualize_correlation(self):
        """Generates a correlation matrix for numeric columns."""
        numeric_columns = self.get_numeric_columns()
        numeric_df = self.df[numeric_columns]     
        correlation_matrix = numeric_df.corr()
        plt.figure(figsize=(12, 8))
        mask = np.zeros_like(correlation_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(correlation_matrix, mask=mask, square=True, linewidths=.5, annot=False, cmap=cmap)
        plt.yticks(rotation=0)
        plt.title("Correlation Matrix of All Numeric Variables")
        plt.show()

    def visualize_skewness_histograms(self, numeric_features):
        """Generates a histogram for the columns to visualise the data skew"""
        numeric_columns = self.get_numeric_columns()                
        sns.set(font_scale=0.7)
        melted = pd.melt(self.df, value_vars=numeric_columns)
        g = sns.FacetGrid(melted, col="variable", col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", stat='density', kde=True)
        plt.show()

    def qq_plot(self, column):
        """
        QQ plot to assess how normally distributed a numeric column is.
        
        Args:
            column (str): The name of the column to visualize.
        """
        qqplot(self.df[column], scale=1, line='q', fit=True)
        plt.title(f"Q-Q Plot of {column}")
        plt.show()

    def visualize_log_transformation(self, column):
        """
        Visualizes the effect of log transformation on a column.
        
        Args:
            column (str): The name of the column to visualize.      
        """
        log_transformed = self.df[column].map(lambda x: np.log(x) if x > 0 else 0)
        sns.histplot(log_transformed, label=f"Skewness: {log_transformed.skew():.2f}", kde=True)
        plt.legend()
        plt.title(f"Log Transformed Histogram of {column}")
        plt.show()

class FailureAnalysis:
    """
    A class to analyze machine failures and recommend operating ranges to minimize failures.
    
    Attributes:
        data (pd.DataFrame): The DataFrame containing the dataset for analysis.
    """    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the class with the DataFrame.
        Args:
            data (pd.DataFrame): The DataFrame containing the dataset for analysis.             
        """
        self.data = data

    def get_numeric_features(self):
        """Return the list of numeric features used in the analysis."""
        return ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    def get_failure_causes(self):
        """Return the list of failure causes used in the analysis."""
        return ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    def preprocess_failure_columns(self):
        """Preprocess failure columns to ensure consistency in analysis."""
        failure_columns = self.get_failure_causes() + ['Machine failure']
        for column in failure_columns:
            if self.data[column].dtypes not in ['int64', 'float64']:
                self.data[column] = self.data[column].replace(
                    {"Failure": 1, "Non-failure": 0}).astype('int')

    def generate_operating_ranges(self):
        """Generate and display the operating ranges for numeric features."""
        numeric_features = self.get_numeric_features()
        ranges_table = self.data[numeric_features].agg(['min', 'max']).T
        ranges_table.columns = ['Min Value', 'Max Value']
        print("Operating Ranges Table:")
        print(ranges_table)

        quality_ranges = self.data.groupby('Type')[numeric_features].agg(['min', 'max'])
        print("Operating Ranges by Product Quality:")
        print(quality_ranges)

    def plot_tool_wear_distribution(self):
        """Analyze the failure data, calculating the number and percentage of failures."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Tool wear [min]'], bins=30, kde=False)
        plt.title('Tool Wear Distribution')
        plt.xlabel('Tool Wear [min]')
        plt.ylabel('Number of Tools')
        plt.axvline(self.data['Tool wear [min]'].max(), color='red', linestyle='--', label='Max Tool Wear')
        plt.legend()
        plt.show()

    def analyze_failures(self):
        """Analyze the failure data, calculating the number and percentage of failures."""
        self.preprocess_failure_columns()
        failure_column = self.data['Machine failure']
        num_failures = failure_column.sum()
        failure_percentage = (num_failures / len(failure_column)) * 100
        print(f"Number of failures: {num_failures}")
        print(f"Percentage of failures: {failure_percentage:.2f}%")

    def plot_failure_causes(self):
        """Plot the number of failures by cause as a bar chart."""
        self.preprocess_failure_columns()
        failure_causes = self.get_failure_causes()
        failure_cause_counts = self.data[failure_causes].sum()
        failure_cause_counts.sort_values(ascending=False, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=failure_cause_counts.index, y=failure_cause_counts.values)
        plt.title('Failures by Cause')
        plt.ylabel('Number of Failures')
        plt.xlabel('Failure Cause')
        plt.show()

    def analyze_failure_correlations(self):
        """
        Analyze correlations between numeric features and failure causes.

        Displays correlation values and bar charts for each failure cause.
        """
        self.preprocess_failure_columns()
        numeric_features = self.get_numeric_features()
        failure_causes = self.get_failure_causes()
        failure_correlation = {}
        for cause in failure_causes:
            failure_rows = self.data[self.data[cause] == 1]
            correlations = failure_rows[numeric_features].corrwith(failure_rows['Tool wear [min]'])
            failure_correlation[cause] = correlations

            print(f"Correlations for {cause}:")
            print(correlations)

            plt.figure(figsize=(10, 6))
            correlations.plot(kind='bar', title=f"Correlations with {cause}", color='teal')
            plt.ylabel('Correlation Coefficient')
            plt.xlabel('Features')
            plt.show()

    def recommend_operating_ranges(self):
        """
        Recommend safe operating ranges for numeric features to minimize failures.      
        """
        risk_thresholds = {
            'Air temperature [K]': (295, 305),
            'Process temperature [K]': (305, 320),
            'Rotational speed [rpm]': (1200, 1500),
            'Torque [Nm]': (30, 50),
            'Tool wear [min]': (0, 250), }        
        print("Recommended Operating Ranges to Minimize Failures:")
        for feature, (lower, upper) in risk_thresholds.items():
            print(f"{feature}: {lower} to {upper}")

# Instantiate classes and process data
info = DataFrameInfo(failure_df)
transform = DataTransform(failure_df)
plotter = Plotter(failure_df)
fail = FailureAnalysis(failure_df)

# Save original DataFrame as a baseline
transform.save_copy("original")

# Inspect initial dataset properties
info.df_info()
info.obtain_shape()
print("Column types:")
print(info.column_types())
print("Descriptive statistics:")
print(info.df_description())
print("Missing values:")
print(info.total_missing())
print("Percentage missing:")
print(info.percentage_missing())

# Delete unnecessary columns for the analysis
columns_to_delete = ["UDI", "Product ID"]
if all(col in transform.df.columns for col in columns_to_delete):
    transform.delete_irrelevant_columns(columns_to_delete)
else:
    print(f"Some columns in {columns_to_delete} were not found in the DataFrame.")

# Replace blank spaces with NaN
print("Before replacing blanks:")
print(transform.df.head())
transform.replace_blank_with_nan()
print("Before replacing blanks:")
print(transform.df.head())

# Fill missing values with median
print("Before filling missing values:")
print(transform.df.isnull().sum())
columns_to_fill = ["Air temperature [K]", "Process temperature [K]"]
transform.fill_missing_with_median(columns_to_fill)
print("After filling missing values:")
print(transform.df.isnull().sum())

# Check nulls before and after handling
print("Nulls before removing missing:")
plotter.visualize_nulls()
print("Before removing missing values:")
print(transform.df.isnull().sum())
transform.remove_missing(["Tool wear [min]"])
print("Nulls after removing missing:")
plotter.visualize_nulls()
print("After removing missing values:")
print(transform.df.isnull().sum())

# Apply column transformations
transform.transform_type_column_values()
transform.transform_failure_column_values()
transform.convert_columns_to_categorical()

# Save snapshot after cleaning and transformations
transform.save_copy("after_cleaning")

# Visualize skew and histograms
plotter.visualize_skew()
numeric_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
plotter.visualize_skewness_histograms(numeric_features)

# Skewness and log transformations
for column in numeric_features:
    skewness = failure_df[column].skew()
    print(f"Skewness of {column}: {skewness}")
    if abs(skewness) > 1:
        print(f"Performing log transformation on {column}")
        plotter.visualize_log_transformation(column)

# Visualize correlation matrix before and after transformations
plotter.visualize_correlation()
transform.transform_skewed_columns()
transform.remove_highly_correlated()
plotter.visualize_correlation()

# Save after transforming skewed columns and highly correlated
transform.save_copy("after_skew_transformations")

# Visualize and handle outliers
for column in numeric_features:
    plotter.visualize_outliers(column)
transform.handle_outliers()
for column in numeric_features:
    plotter.visualize_outliers(column)

# Save final DataFrame
transform.save_copy("final")

# Comparison of original and final DataFrame
original_df = transform.snapshots["original"]
final_df = transform.snapshots["final"]

print(f"Original shape: {original_df.shape}, Final shape: {final_df.shape}")
print("Descriptive statistics of original and final DataFrame:")
print("Original:")
print(original_df.describe())
print("Final:")
print(final_df.describe())

# Failure tests section

# Operating ranges
fail.generate_operating_ranges()

# Tool wear distribution
fail.plot_tool_wear_distribution()

# Failure analysis
fail.analyze_failures()
fail.plot_failure_causes()

# Correlation analysis
fail.analyze_failure_correlations()

# Recommended ranges
fail.recommend_operating_ranges()
