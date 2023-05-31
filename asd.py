import json
import torch
import torch.nn as nn
contents=[]
labels=[]
with open('文本分类/data/train.json','r',encoding='utf-8') as f:
        for dict in json.load(f):
            contents.append(dict['text'])
            labels.append(dict['label'])
classes_2_index={"label":0}
for label in labels:
    classes_2_index[label]=classes_2_index.get(label,len(label))




print(classes_2_index)