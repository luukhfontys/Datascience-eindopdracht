import pandas as pd
import numpy as np
import scipy.stats as sp
import re
import os
from tqdm import tqdm

def load_data_index(data_path: str, bearing: int, index: int) -> pd.DataFrame:
    """
    Load and filter data for a specific bearing and index from a CSV file.

    Parameters:
    - data_path: str, the path to the directory containing the data files.
    - bearing: int, the bearing number to filter columns by.
    - index: int, the index of the CSV file to load.

    Returns:
    - pd.DataFrame: DataFrame containing the filtered data.
    """
    index_file_path = f'{data_path}/{index}.csv'
    df_csv = pd.read_csv(index_file_path, delimiter=';')

    df_csv_filtered = df_csv[[col for col in df_csv.columns if str(bearing) in col]]
    return df_csv_filtered

def spectral_flatness(x: np.ndarray) -> float:
    """
    Calculate the spectral flatness of a signal.

    Parameters:
    - x: np.ndarray, the input signal.

    Returns:
    - float: The spectral flatness value.
    """
    fft_spectrum = np.fft.fft(x)
    power_spectrum = np.abs(fft_spectrum)**2

    # Vermijd nul waardes om delen door 0 te voorkomen
    power_spectrum += 1e-10

    log_power_spectrum = np.log(power_spectrum)
    Geometric_mean = np.exp(np.mean(log_power_spectrum))
    arithmetic_mean = np.mean(power_spectrum)

    spectral_flatness_result = Geometric_mean / arithmetic_mean

    return spectral_flatness_result



def get_features(df: pd.DataFrame):
    features = {
        'mean_x': df.iloc[:, 0].mean(),
        'mean_y': df.iloc[:, 1].mean(),
        'std_x': df.iloc[:, 0].std(),
        'std_y': df.iloc[:, 1].std(),
        'rms_x': (sum(df.iloc[:, 0]**2)/len(df.iloc[:, 0]))**0.5,
        'rms_y': (sum(df.iloc[:, 1]**2)/len(df.iloc[:, 1]))**0.5,
        'kurtosis_x': sp.kurtosis(df.iloc[:, 0]),
        'kurtosis_y': sp.kurtosis(df.iloc[:, 1]),
        'variance_x': df.iloc[:, 0].var(),
        'variance_y': df.iloc[:, 1].var(),
        'crest_factor_x': np.max(np.abs(df.iloc[:, 0])) / (sum(df.iloc[:, 0]**2)/len(df.iloc[:, 0]))**0.5,
        'crest_factor_y': np.max(np.abs(df.iloc[:, 1])) / (sum(df.iloc[:, 1]**2)/len(df.iloc[:, 1]))**0.5,
        'skewness_x': sp.skew(df.iloc[:, 0]),
        'skewness_y': sp.skew(df.iloc[:, 1]),
        'spectral_flatness_x': spectral_flatness(df.iloc[:, 0]),
        'spectral_flatness_y': spectral_flatness(df.iloc[:, 1])
    }
    return features

def find_highest_number_in_filenames(folder_path):
    files = os.listdir(folder_path)
    pattern = re.compile(r'^(\d+)\.csv$')
    numbers = [int(pattern.match(file).group(1)) for file in files if pattern.match(file)]
    return max(numbers) if numbers else None

def generate_dataset(data_path: str):
    bearing = 4
    data_length = find_highest_number_in_filenames(data_path)
    if data_length is None:
        return pd.DataFrame()
    
    dataframe_list = []
    
    for index in tqdm(range(0, data_length + 1), desc="Processing files"):
        df = load_data_index(data_path, bearing, index)
        features = get_features(df)
        features['index'] = index
        dataframe_list.append(pd.DataFrame(features, index=[index]))
    dataset = pd.concat(dataframe_list)
    dataset.set_index('index', inplace=True)
    return dataset