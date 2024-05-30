from functions import *

def main():
    # opg 1a
    data_path = 'data'
    plot_ex_1a(data_path)

    # opg 1b
    dataset = generate_dataset(data_path)
    dataset.to_excel('dataset_features.xlsx')

    

main()