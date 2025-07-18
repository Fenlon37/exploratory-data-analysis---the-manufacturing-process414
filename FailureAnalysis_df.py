from shared_imports import *

"""File path and initial data load"""
file_path = "failure_data.csv"
failure_df = obtain_local_data(file_path)

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
        tool_wear = self.data['Tool wear [min]'].to_numpy()
        sns.histplot(tool_wear, bins=30, kde=False)
        plt.title('Tool Wear Distribution')
        plt.xlabel('Tool Wear [min]')
        plt.ylabel('Number of Tools')
        plt.axvline(tool_wear.max(), color='red', linestyle='--', label='Max Tool Wear')
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

    def plot_boxplots_by_failure(self):
        """Generate boxplots for numeric features grouped by failure conditions."""
        self.preprocess_failure_columns()
        numeric_features = self.get_numeric_features()

        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data['Machine failure'], y=self.data[feature])
            plt.title(f"{feature} by Failure Condition")
            plt.xlabel("Machine Failure (0 = Non-failure, 1 = Failure)")
            plt.ylabel(feature)
            plt.show()
   

    def analyze_failure_correlations(self):
        """Analyze correlations between numeric features and failure causes."""
        self.preprocess_failure_columns()
        numeric_features = self.get_numeric_features()
        failure_causes = self.get_failure_causes()
        failure_correlation = {}
        for cause in failure_causes:
            failure_rows = self.data[self.data[cause] == 1]
            # Ensure all columns are Series (should be if DataFrame is constructed properly)
            failure_rows_df = pd.DataFrame(failure_rows, index=failure_rows.index)
            correlations = failure_rows_df[numeric_features].corrwith(failure_rows_df['Tool wear [min]'])

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

