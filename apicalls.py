import os
import requests
import json

with open('config.json','r') as f:
    config = json.load(f) 

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')


#Call each API endpoint and store the responses
response1 = requests.post(URL + 'prediction', data={'path': json.dumps(test_data_path)})
response2 = requests.get(URL + 'scoring')
response3 = requests.get(URL + 'summarystats')
response4 = requests.get(URL + 'diagnostics')

#combine all API responses
responses = {
    'prediction': response1.json(),
    'scoring': response2.json(),
    'summarystats': response3.json(),
    'diagnostics': response4.json()
}

#write the responses to your workspace
with open('apireturns2.txt', 'w') as f:
    json.dump(responses, f)

