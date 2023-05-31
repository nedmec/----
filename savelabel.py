import json
import torch.nn as nn
from config import *
import torch
from Block import TextCNNModel
from TextDataset import *
from torch.utils.data import Dataset, DataLoader
classes={'diplomatic_news': 0, 
         'emergency_news': 1, 
         'music_news': 2, 
         'agricultural_news': 3, 
         'stock_news': 4, 
         'game_news': 5, 
         'economic_news': 6, 
         'football_yc': 7, 
         'scientific_news': 8}
def read_data(jsonname):
    contents=[]
    labels=[]
    with open(jsonname,'r',encoding='utf-8') as f:
        for dict in json.load(f):
            contents.append(dict['text'])
            labels.append(dict['label'])
    return contents,labels

def read_data2(jsonname):
    contents=[]
    with open(jsonname,'r',encoding='utf-8') as f:
        for dict in json.load(f):
            contents.append(dict['text'])

    return contents


def built_vocab(train_texts,embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)

if __name__=='__main__':


    train_text, train_label = read_data(Train_sample_path)
    Valid_text = read_data2("./data/test1_unlable.json")
 
    embeddin_num = 64
    max_len = Text_Len
    batch_size = 1
    epoch = 100
    lr = 0.0000001
    channels = 2
    class_num = len(set(train_label))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    word_2_index, words_embedding = built_vocab(train_text, embeddin_num)

   
    Valid_dataset = TextDataset(Valid_text, train_label, word_2_index,max_len)
    Valid_loader = DataLoader(Valid_dataset, batch_size, shuffle=False)
 
    model = TextCNNModel(words_embedding, max_len, class_num, channels).to(device)
    model.load_state_dict(torch.load("./data/model.pth"))
    outclass = {}
    for key,val in classes.items():
        outclass[val]=key
    model.eval()
    right_num = 0
    with open("./data/test1_unlable.json","r",encoding="utf-8") as f:
        f1=json.load(f)
        i=0
        for batch_idx, batch_label in Valid_loader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            pre = outclass[model(batch_idx).item()]
            f1[i]["label"]=pre
            i+=1
        with open("./data/test1_unlable.json","w",encoding="utf-8") as f2:
            str_json = json.dumps(f1, ensure_ascii=False, indent=2)
            f2.write(str_json)
        



