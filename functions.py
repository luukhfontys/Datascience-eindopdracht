import pandas as pd
import numpy as np

def load_data_index(data_path: str, bearing: int, index: int):
    index_file_path = data_path + f'{index}'.csv
    df_csv = pd.read_csv(index_file_path, delimiter=';')
    return df_csv