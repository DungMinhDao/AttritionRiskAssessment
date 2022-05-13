from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


#################Function for training the model
def train_model():
    df = pd.read_csv(dataset_csv_path)
    X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = df['exited']
    #use this logistic regression for training
    lr = LogisticRegression()
    
    #fit the logistic regression to your data
    lr.fit(X, y)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.isdir(config['output_model_path']):
        os.mkdir(config['output_model_path'])
    pickle.dump(lr, open(model_path, 'wb'))

if __name__ == '__main__':
    train_model()