import json 

def read_json(path):
    with open(path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    return data 

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as fout:
        json.dump(data, fout)