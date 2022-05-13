import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    input_df = []
    file_list = []
    for file_name in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file_name)
        currentdf = pd.read_csv(file_path)
        input_df.append(currentdf)
        file_list.append(file_path + '\n')

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    df = pd.concat(input_df, axis=0, ignore_index=True).drop_duplicates()
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.writelines(file_list)


if __name__ == '__main__':
    merge_multiple_dataframe()
