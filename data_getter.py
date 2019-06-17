import requests
import yaml
import json

api_key_file = yaml.safe_load(open('api_keys.yaml'))

data = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSFT&apikey={}'.format(api_key_file['alphakey']))
data = data.json()
print(data)