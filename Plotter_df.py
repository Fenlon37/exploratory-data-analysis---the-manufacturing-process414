from shared_imports import *

"""File path and initial data load"""
file_path = "failure_data.csv"
failure_df = obtain_local_data(file_path)

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
        log_transformed = self.df[column].apply(lambda x: np.log1p(x) if x >= 0 else np.nan)
        sns.histplot(log_transformed, label=f"Skewness: {log_transformed.skew():.2f}", kde=True)
        plt.legend()
        plt.title(f"Log Transformed Histogram of {column}")
        plt.show()

    def visualize_boxcox_transformation(self, column):
        """
        Visualizes the effect of Box-Cox transformation on a column.
        """
        col_data = self.df[column].values
        if col_data.ndim == 1 and (col_data > 0).all():
            result = boxcox(col_data)
            transformed = np.asarray(result[0] if isinstance(result, tuple) else result)
            sns.histplot(transformed, label=f"Box-Cox, Skew: {pd.Series(transformed).skew():.2f}", kde=True)
            plt.legend()
            plt.title(f"Box-Cox Transformed Histogram of {column}")
            plt.show()
        else:
            print(f"Box-Cox transformation cannot be applied to {column} (non-positive values or not 1D)")