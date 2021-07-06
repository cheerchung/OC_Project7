import requests

url = 'http://localhost:5000/results'
# if having a client ID
client = {'id': 100038, 'gender': '', 'amt': '', 'age': '', 'education': '', 'ext3': ''}

# if not having a client ID
# low risk example
client = {'id': '', 'gender': '1', 'amt': '1000000', 'age': '40', 'education': '1', 'ext3': '0.7'}
# high risk example
#client = {'id': '', 'gender': '0', 'amt': '1000000', 'age': '20', 'education': '0', 'ext3': '0.1'}

r = requests.post(url, json=client)

print(r.json())
