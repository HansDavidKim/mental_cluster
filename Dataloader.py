### Load data from csv or parquet files.
import pandas as pd
from enum import Enum

class DataFormat(Enum):
    TITLE_COMMENT = 0
    TITLE_ONLY    = 1
    COMMENT_ONLY  = 2

def load_data(
        file_path: str = './data/Sentiment_Monitoring_2020-01-01.csv',
        ) -> pd.DataFrame:
    ### Parsing file format from input file path.
    file_format = file_path.split('.')[-1].lower()

    if file_format == 'csv':
        ### Load data from csv file.
        data = pd.read_csv(file_path)
    elif file_format == 'parquet':
        ### Load data from parquet file.
        data = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Supported formats are 'csv' and 'parquet'.")
    
    ### Return loaded data as a pandas DataFrame.
    return data

def preprocess_data(
        data: pd.DataFrame, 
        data_format: DataFormat
        ) -> pd.DataFrame:
    
    if data_format == DataFormat.TITLE_COMMENT:
        data['sequence'] = '[TITLE] ' + data['videoTitle'] + ' [COMMENT] ' + data['text']

    elif data_format == DataFormat.TITLE_ONLY:
        data['sequence'] = '[TITLE] ' + data['videoTitle']

    elif data_format == DataFormat.COMMENT_ONLY:
        data['sequence'] = '[COMMENT] ' + data['text']

    else:
        raise ValueError(f"Unsupported data format: {data_format}. Supported formats are TITLE_COMMENT, TITLE_ONLY, and COMMENT_ONLY.")
    
    ### We don't have to drop the original columns, but we can keep them as they are needed.
    data = data[['publishedAt', 'videoTitle', 'text', 'sequence']]
    return data
