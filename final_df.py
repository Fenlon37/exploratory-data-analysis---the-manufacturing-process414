from DataFrameInfo_df import DataFrameInfo
from DataTransform_df import DataTransform
from FailureAnalysis_df import FailureAnalysis
from loading_dataframe import obtain_local_data
from Plotter_df import Plotter
from shared_imports import *

"""File path and initial data load"""
file_path = "failure_data.csv"
failure_df = obtain_local_data(file_path)

# Instantiate classes and process data
info = DataFrameInfo(failure_df)
transform = DataTransform(failure_df)
plotter = Plotter(failure_df)
fail = FailureAnalysis(failure_df)

# Save original DataFrame as a baseline
transform.save_copy("original")
print("\n--- Snapshot: original ---")
print(transform.snapshots["original"].head())

# Inspect initial dataset properties
info.data_frame_full_info()

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
print("After replacing blanks:")
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
print("\n--- Snapshot: after_cleaning ---")
print(transform.snapshots["after_cleaning"].head())

# Visualize skew and histograms
plotter.visualize_skew()
numeric_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
plotter.visualize_skewness_histograms(numeric_features)

# Skewness and log transformations
for column in numeric_features:
    skewness = transform.df[column].skew()  # Use transform.df instead of failure_df
    print(f"Skewness of {column}: {skewness}")
    if abs(skewness) > 1:
        print(f"Assessing log transformation of {column}")
        plotter.visualize_log_transformation(column)
        print(f"Assessing Box-Cox transformation of {column}")
        plotter.visualize_boxcox_transformation(column)
        
# Box Cox transformation
for column in numeric_features:
    if transform.df[column].skew() > 1:  # Use transform.df
        transform.boxcox_transform(column)

# Update plotter to use transformed data
plotter.df = transform.df

# Visualize Box-Cox transformed data
for column in numeric_features:
    print(f"Visualizing Box-Cox transformed data for {column}")
    plotter.visualize_boxcox_transformation(column)

# Skewness assessment after Box Cox transformation
for column in numeric_features:
    skewness = failure_df[column].skew()
    print(f"Skewness of {column}: {skewness}")

# Plot boxplots for numeric features grouped by Failure or Non-failure
fail.plot_boxplots_by_failure()

# Visualize correlation matrix before and after transformations
plotter.visualize_correlation()
transform.transform_skewed_columns()
transform.remove_highly_correlated()
plotter.visualize_correlation()

# Save after transforming skewed columns and highly correlated
transform.save_copy("after_skew_transformations")
print("\n--- Snapshot: after_skew_transformations ---")
print(transform.snapshots["after_skew_transformations"].head())

# Visualise QQ plots
for column in numeric_features:
    plotter.qq_plot(column)

# Visualize and handle outliers
for column in numeric_features:
    plotter.visualize_outliers(column)
transform.handle_outliers()
for column in numeric_features:
    plotter.visualize_outliers(column)

# Save final DataFrame
transform.save_copy("final")
print("\n--- Snapshot: final ---")
print(transform.snapshots["final"].head())

# Comparison of original and final DataFrame
original_df = transform.snapshots["original"]
final_df = transform.snapshots["final"]

print("\n--- Original DataFrame (shape) ---")
print(original_df.shape)
print("\n--- Final DataFrame (shape) ---")
print(final_df.shape)

print("\n--- Original DataFrame (describe) ---")
print(original_df.describe())
print("\n--- Final DataFrame (describe) ---")
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
