import os
import subprocess
import json
import pandas as pd
import training
import scoring
import deployment
import diagnostics
import reporting

with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
dataset_csv_path = os.path.join(output_folder_path, 'finaldata.csv')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

deploy_path = config['prod_deployment_path']
ingested_files_path = os.path.join(deploy_path, 'ingestedfiles.txt')
model_path = os.path.join(deploy_path, 'trainedmodel.pkl')
latest_score = os.path.join(deploy_path, 'latestscore.txt')

##################Check and read new data
#first, read ingestedfiles.txt
with open(ingested_files_path) as f:
    ingested_files_list = f.read().splitlines()
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
all_files_exist = True
for file_name in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, file_name)
    if file_path not in ingested_files_list:
        all_files_exist = False
        break

if not all_files_exist:
    print("New data found. Ingest new datasets from {}".format(input_folder_path))
    subprocess.call(['python', 'ingestion.py'])


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if all_files_exist:
    print("No new data found. Exit the process")
    quit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(latest_score, 'r') as f:
    latest_score = float(f.read())
new_score = scoring.score_model(model_path, dataset_csv_path)
print('lastest score: {}, score on newly ingested data: {}'.format(latest_score, new_score))

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_score >= latest_score:
    print('No model drift happended')
    quit()
else:
    print('Detect model drift. Retrain and redeploy model')

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
subprocess.call(['python', 'training.py'])
subprocess.call(['python', 'scoring.py'])
subprocess.call(['python', 'deployment.py'])

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.call(['python', 'diagnostics.py'])
subprocess.call(['python', 'reporting.py'])






