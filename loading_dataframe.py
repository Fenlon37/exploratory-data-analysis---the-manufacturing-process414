import pandas as pd

def obtain_local_data(file_path):
 try:
   failure_df = pd.read_csv(file_path)
   shape = failure_df.shape
   print(f'This dataset has {shape[0]} rows and {shape[1]} columns')
   print(failure_df.head(5))
   failure_df.info()
   failure_df.describe()
   return failure_df
 except FileNotFoundError:
   raise Exception(f"File not found at {file_path}. Please provide the correct file path.")
 except Exception as e:
   raise Exception(f"An error occurred while loading the data: {e}")

if __name__ == "__main__":
    # Path to the local CSV file
    file_path = "failure_data.csv"
    # Load the data
    failure_df = obtain_local_data(file_path)
