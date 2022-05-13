from flask import Flask, request
import json
import os
import scoring
import diagnostics


######################Set up variables for use in our script
app = Flask(__name__)
# app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
prediction_model = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    test_data_path = request.form.get('path')
    result = diagnostics.model_predictions(prediction_model, test_data_path[1:-1])
    return json.dumps([int(item) for item in result])

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1_score = scoring.score_model(prediction_model, test_data_path)
    return json.dumps(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary():        
    #check means, medians, and standard deviations for each column
    df_stat = diagnostics.dataframe_summary(dataset_csv_path)
    return df_stat.to_dict()

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    #check timing and percent NA values
    timing = diagnostics.execution_time()
    na_percents = diagnostics.dataframe_missing(dataset_csv_path)
    if os.path.isfile('dependencies.json'):
        dependencies = json.load(open('dependencies.json'))
    else:
        dependencies = diagnostics.dependencies_checking().to_dict('records')
    diagnose_dict = {
        'timing': timing,
        'na_percents': na_percents,
        'dependencies': dependencies
    }
    return json.dumps(diagnose_dict)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
