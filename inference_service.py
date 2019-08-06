import requests
import json

endpoint = 'http://127.0.0.1:5000/'

input_data = json.dumps({'data': [
    [8700,11080,35000],
    [10800,11090,25000]
]})

resp = requests.post(endpoint, input_data)
print(resp.text)
