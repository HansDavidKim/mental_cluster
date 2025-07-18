### This part of the code is used to load data from a file and preprocess it 
### for more sophisticated analysis.
###
### You can specify the file format (CSV or Parquet) and whether to preprocess data or not.

import os
import pandas as pd
import re
from enum import Enum

CSV = 'csv'
PARQUET = 'parquet'

def load_data(file_name: str, file_format: str = CSV) -> pd.DataFrame:
    """
    Load data from a file and return it as a pandas DataFrame.
    
    :param file_path: Path to the data file.
    :param file_format: Format of the data file ('csv' or 'parquet').
    :return: Loaded data as a pandas DataFrame.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to DataLoad.py
    file_path = os.path.join(base_dir, '..', 'data', file_name)

    if file_format.lower() == CSV:
        return pd.read_csv(file_path)
    elif file_format.lower() == PARQUET:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Supported formats are 'csv' and 'parquet'.")
    
### Preprocess the video title by removing unnecessary meta information.
def preprocess_video_title(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the video title by removing meta information, dates, news agencies, and irrelevant tags.

    :param data: DataFrame with a 'videoTitle' column.
    :return: DataFrame with 'cleanTitle'.
    """

    # Apply to DataFrame
    data = data.copy()
    data['comment_only'] = '[COMMENT] ' + data['cleanText']
    data['title_comment'] = '[TITLE] ' + data['title'] + ' [COMMENT] ' + data['text']
    return data

if __name__ == "__main__":
    # Example usage
    df = load_data('Sentiment_Monitoring_2020-01-01.csv', CSV)
    df = preprocess_video_title(df)
    print(df.head())