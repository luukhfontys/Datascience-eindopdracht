from functions import *

def main():
    data_path = 'data'
    plot_ex_1a(data_path)

    dataset = generate_dataset(data_path)
    dataset.to_excel('dataset_features.xlsx')

main()