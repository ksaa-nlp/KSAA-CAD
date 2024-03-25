import os
import numpy as np
from utils import read_json, write_json

submission_template = "outputs/task2/{}/revdict-preds-combined-cos.json"

all_electra = []
all_sgns = []
for model in ["marbert", "camelbert-msa"]:
    filepath = submission_template.format(model)
    if not os.path.exists(filepath):
        print(f"> Skipping {filepath}")
        continue

    print(f"> Reading {filepath}")
    data = read_json(filepath)
    electra = []
    sgns = []
    for row in data:
        electra += [np.array(row["electra"])]
        sgns += [np.array(row["sgns"])]
    all_electra += [np.array(electra)]
    all_sgns += [np.array(sgns)]

electra = np.array(all_electra).mean(axis=0)
sgns = np.array(all_sgns).mean(axis=0)

print(electra.shape)
print(sgns.shape)

for i in range(len(data)):
    data[i]["electra"] = list(electra[i])
    data[i]["sgns"] = list(sgns[i])

write_json("outputs/task2/camelbert_msa_marbert.json", data)