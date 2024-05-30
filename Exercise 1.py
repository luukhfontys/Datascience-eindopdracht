from functions import *

def main():
    dataset = generate_dataset('train')
    dataset.to_excel('dataset_features.xlsx')

main()