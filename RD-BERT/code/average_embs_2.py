import os
import numpy as np
from glob import glob 
from utils import read_json, write_json

files = glob("outputs/task1/*/revdict-preds-combined-cos.json")

all_sgns = []
all_electra = []
all_bertmsa=[]
all_bertseg=[]

for filepath in files:
    modelname = filepath.split("/")[2]
    if modelname not in ["camelbert-msa", "marbertv2"]:
        continue
    print(f"Model: {modelname}")
    data = read_json(filepath)
    electra = []
    sgns = []
    bertmsa=[]
    bertseg=[]
    for row in data:
        electra += [np.array(row["electra"])]
        sgns += [np.array(row["sgns"])]
        bertmsa += [np.array(row["bertmsa"])]
        bertseg += [np.array(row["bertseg"])]
    all_electra += [np.array(electra)]
    all_sgns += [np.array(sgns)]
    all_bertmsa += [np.array(bertmsa)]
    all_bertseg += [np.array(bertseg)]
    

electra = np.array(all_electra).mean(axis=0)
sgns = np.array(all_sgns).mean(axis=0)
bertmsa = np.array(all_bertmsa).mean(axis=0)
bertseg = np.array(all_bertseg).mean(axis=0)

for i in range(len(data)):
    data[i]["electra"] = list(electra[i])
    data[i]["sgns"] = list(sgns[i])
    data[i]["bertseg"] = list(bertseg[i])
    data[i]["bertseg"] = list(bertseg[i])

# python code/score.py --submission_path outputs/task1/ensemble_cos_final.json --reference_files_dir data/ar.dev.json
write_json(f"outputs/task1/ensemble_cos_final.json", data)