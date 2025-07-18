from shared_imports import *

"""File path and initial data load"""
file_path = "failure_data.csv"
failure_df = obtain_local_data(file_path)

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

    def boxcox_transform(self, column):
        """
        Apply the Box-Cox transformation to a specified column.
        """
        if column in self.df.columns:
            col_data = self.df[column].values
            if col_data.ndim == 1 and (col_data > 0).all():
                transformed = boxcox(col_data)[0]
                self.df[column] = pd.Series(transformed, index=self.df.index)
                print(f"Applied Box-Cox transformation to column: {column}")
            else:
                print(f"Box-Cox transformation cannot be applied to {column} (non-positive values or not 1D)")
        else:
            print(f"Box-Cox transformation cannot be applied to {column} (column not found)")

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
