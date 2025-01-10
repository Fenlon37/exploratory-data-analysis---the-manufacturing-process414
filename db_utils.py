from sqlalchemy import create_engine
import pandas as pd
import yaml

class RDSDatabaseConnector:
  def __init__(self, credentials):
    """ Initialise the RDS instance with the credentials"""
    self.credentials = credentials
    self.engine = None

  def load_credentials(self, file_path='credentials.yaml'):
    try:
     with open(file_path, 'r') as file:
      return yaml.safe_load(file)
    except FileNotFoundError:
     raise Exception(f"Credentials file not found at {file_path}")
    except yaml.YAMLError as e:
     raise Exception(f"Error parsing YAML file: {e}")
  
  def init_engine(self):
    try:
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.credentials['RDS_HOST']
        USER = self.credentials['RDS_USER']
        PASSWORD = self.credentials['RDS_PASSWORD']
        DATABASE = self.credentials['RDS_DATABASE']
        PORT = self.credentials['RDS_PORT']
        self.engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
    except KeyError as e:
        raise Exception(f"Missing required credential: {e}")
    except Exception as e:
        raise Exception(f"Error initializing engine: {e}")
  
  def extract_data(self, table_name="failure_data"):
    try:
      extracted = f"SELECT * FROM {table_name}"
      df = pd.read_sql(extracted, self.engine)
      print(f"Data extracted successfully from table: {table_name}")
      return df
    except Exception as e:
      raise Exception(f"Error extracting data from table {table_name}: {e}")

  def save_to_csv(self, df, file_path="failure_data.csv"):
    try:
     df.to_csv(file_path, index=False)
     print(f"Data saved successfully to {file_path}")
    except Exception as e:
     raise Exception(f"Error saving data to CSV: {e}")

from db_utils import RDSDatabaseConnector
if __name__ == "__main__":
    # Step 1: Initialise connector and load credentials
    connector = RDSDatabaseConnector({})
    credentials = connector.load_credentials()
    connector.credentials = credentials

    # Step 2: Initialise the SQLAlchemy engine
    connector.init_engine()

    # Step 3: Extract data to DataFrame
    failure_data_df = connector.extract_data()

    # Step 4: Save data to CSV
    connector.save_to_csv(failure_data_df, "failure_data.csv")
