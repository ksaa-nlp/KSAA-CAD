import os
import numpy as np
from utils import read_json, write_json

submission_template = "outputs/task2/submission-{}/revdict-preds-combined-{}.json"

path = "outputs/task2/revdict-preds-alignment-sgns.json"

print(electra.shape)
print(sgns.shape)

for i in range(len(data)):
    data[i]["electra"] = list(electra[i])
    data[i]["sgns"] = list(sgns[i])

write_json("outputs/task2/rashid_task2_test_7.json", data)