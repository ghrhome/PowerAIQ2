import json

with open('train_data.json') as data_file:
    data = json.load(data_file)

print(len(data))
