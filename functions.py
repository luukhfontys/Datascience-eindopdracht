import pandas as pd
import numpy as np
import re

def load_data_index(data_path: str, bearing: int, index: int):
    index_file_path = f'{data_path}/{index}.csv'
    df_csv = pd.read_csv(index_file_path, delimiter=';')

    df_csv_filtered = df_csv[[col for col in df_csv.columns if str(bearing) in col]]
    return df_csv_filtered

def get_features(df: str):
    mean = [df.iloc[:, 0].mean(), df.iloc[:, 1].mean()]
    std = [df.iloc[:, 0].std(), df.iloc[:, 1].std()]
    rms = [(sum(df.iloc[:, 0]**2)/len(df.iloc[:, 0]))**0.5, (sum(df.iloc[:, 1]**2)/len(df.iloc[:, 1]))**0.5]
    variance = [df.iloc[:, 0].var(), df.iloc[:, 1].var()]
    crest_factor = [np.max(np.abs(df.iloc[:, 0])) / rms[0], np.max(np.abs(df.iloc[:, 1])) / rms[1]]
    
    return mean + std + rms



df = load_data_index('train', 1, 0)
test = get_features(df)
x=1