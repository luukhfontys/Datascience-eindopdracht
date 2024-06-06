import pandas as pd
from functions import *

def Ex1ab():
    # opg 1a
    data_path = 'data'
    plot_ex_1a(data_path, False)

    # opg 1b
    dataset = generate_dataset(data_path)
    dataset.to_excel('dataset_features.xlsx')

def Ex1c():
    df = pd.read_excel('dataset_features.xlsx')
    plot_scatter_matrix(df)

#Ex1ab()

Ex1c()
