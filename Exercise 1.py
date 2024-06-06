import pandas as pd
from functions import *

def Ex1ab():
    # opg 1a
    data_path = 'data'
    plot_ex_1a(data_path, False)

    # opg 1b
    dataset = generate_dataset(data_path)
    # dataset.to_excel('dataset_features.xlsx')
    corr_matrix = dataset.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()
    

main()