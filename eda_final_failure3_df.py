from statsmodels.graphics.gofplots import qqplot
from loading_dataframe import obtain_local_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""File path and initial data load"""
file_path = "failure_data.csv"
failure_df = obtain_local_data(file_path)

"""DataFrameInfo class for inspecting the DataFrame"""
class DataFrameInfo:
    def __init__(self, df):
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
        """Returns descriptive statistics for the DataFrame, ignoring nulls."""
        return self.df.describe()

    def obtain_shape(self):
        """Prints and returns the column and row details for the DataFrame."""
        shape = self.df.shape
        print(f"This dataset has {shape[0]} rows and {shape[1]} columns")
        return shape

    def distinct_counts(self):
        """Returns counts of distinct values for categorical columns."""
        return self.df.select_dtypes(include=['category', 'object']).nunique()
   
    def extract_statistics(self):
        """Median, mean, and standard deviation for numeric columns."""
        return pd.DataFrame({
            'median': self.df.median(numeric_only=True),
            'std': self.df.std(numeric_only=True),
            'mean': self.df.mean(numeric_only=True)
        })

    def percentage_missing(self):
        """Returns the percentage of nulls in each column."""
        return (self.df.isnull().sum() / len(self.df)) * 100

"""DataTransform class for cleaning and manipulating data of the dataframe"""
class DataTransform:
    def __init__(self, df):
        self.df = df
        self.snapshots = {}

    def delete_irrelevant_columns(self, columns):
        """Deleting columns not useful within analysis"""
        self.df = self.df.drop(columns=columns)

    def save_copy(self, label):
        """Saves a copy of the current DataFrame with a label for comparison."""
        self.snapshots[label] = self.df.copy()
        print(f"Snapshot '{label}' saved.")

    def replace_blank_with_nan(self):
        """Replaces the empty cells with NaN"""
        self.df = self.df.replace({" ": np.nan, "": np.nan}, regex=True)
        print("Null values after replacing blanks with NaN:")
        print(self.df.isnull().sum())
    
    def fill_missing_with_median(self, columns):
        """Replaces NaN cells with the median of respective column"""
        for column in columns:
            if column in self.df.columns:
                median_value = self.df[column].median()
                self.df[column] = self.df[column].fillna(median_value)

    def remove_missing(self, subset):
        """Drops remaining rows still containing NaN cells""" 
        before_rows = len(self.df)
        self.df = self.df.dropna(subset=subset)
        after_rows = len(self.df)
        print(f"Dropped {before_rows - after_rows} rows due to missing values.")  

    def transform_column_values(self):
        """Transforming "Type" column values."""
        if "Type" in self.df.columns:
            self.df["Type"] = self.df["Type"].replace({"H": "High", "M": "Medium", "L": "Low"})

        # Transforming columns 9-14 that are of Boolean format to "Non-failure" and "Failure"
        failure_columns = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        self.df[failure_columns] = self.df[failure_columns].replace({0: "Non-failure", 1: "Failure"})

    def convert_columns_to_categorical(self):
        """Converting failure columns 9-14 to categorical type"""
        failure_columns = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]        
        self.df[failure_columns] = self.df[failure_columns].astype('category')

    def transform_skewed_columns(self, threshold=1):
        """Applies transformations to skewed numeric columns."""
        numeric_columns = self.get_numeric_columns()
        for column in numeric_columns:            
            skewness = self.df[column].skew()
            print(f"Skewness of {column}: {skewness}")
            if abs(skewness) > threshold:
                self.df[column] = np.log1p(self.df[column])

    def handle_outliers(self, z_thresh=3):
        """Removes rows with outliers based on Z-scores."""
        numeric_columns = self.get_numeric_columns()      
        for column in numeric_columns:
            z_scores = (self.df[column] - self.df[column].mean()) / self.df[column].std()
            self.df = self.df[z_scores.abs() < z_thresh]

    def remove_highly_correlated(self, threshold=0.9):
        """Removes any highly correlated columns."""
        numeric_columns = self.get_numeric_columns()
        numeric_df = self.df[numeric_columns]  
        corr_matrix = numeric_df.corr()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.df = self.df.drop(columns=to_drop)

    def get_numeric_columns(self):
        """Returns numeric columns explicitly excluding categorical and deleted."""
        exclude_columns = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_columns if col not in exclude_columns]    

# Plotter class for displaying patterns within the data.
class Plotter:
    def __init__(self, df):
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
        """Generates a correlation heatmap."""
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
        """QQ plot to assess how normally distributed column is"""
        qqplot(self.df[column], scale=1, line='q', fit=True)
        plt.title(f"Q-Q Plot of {column}")
        plt.show()

    def visualize_log_transformation(self, column):
        """Visualizes the effect of log transformation on a column."""
        log_transformed = self.df[column].map(lambda x: np.log(x) if x > 0 else 0)
        sns.histplot(log_transformed, label=f"Skewness: {log_transformed.skew():.2f}", kde=True)
        plt.legend()
        plt.title(f"Log Transformed Histogram of {column}")
        plt.show()



# Instantiate classes and process data
info = DataFrameInfo(failure_df)
transform = DataTransform(failure_df)
plotter = Plotter(failure_df)

# Save original DataFrame as a baseline
transform.save_copy("original")

# Delete unnecessary columns for the analysis
columns_to_delete = ["UDI", "Product ID"]
if all(col in transform.df.columns for col in columns_to_delete):
    transform.delete_irrelevant_columns(columns_to_delete)
else:
    print(f"Some columns in {columns_to_delete} were not found in the DataFrame.")

# Replace blank spaces
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

# Apply column transformations
transform.transform_column_values()
transform.convert_columns_to_categorical()

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
