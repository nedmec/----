import json
import torch.nn as nn
from config import *
import torch
from Block import TextCNNModel
from TextDataset import *
from torch.utils.data import Dataset, DataLoader
def read_data(jsonname):
    contents=[]
    labels=[]
    with open(jsonname,'r',encoding='utf-8') as f:
        for dict in json.load(f):
            contents.append(dict['text'])
            labels.append(dict['label'])
    return contents,labels


def built_vocab(train_texts,embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)

if __name__=='__main__':
    train_text, train_label = read_data(Train_sample_path)
    Valid_text, Valid_label = read_data(Valid_sample_path)
 
    embeddin_num = 64
    max_len = Text_Len
    batch_size = 1
    epoch = 100
    lr = 0.0000001
    channels = 2
    class_num = len(set(train_label))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    word_2_index, words_embedding = built_vocab(train_text, embeddin_num)

    train_dataset = TextDataset(train_text, train_label, word_2_index,max_len)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    Valid_dataset = TextDataset(Valid_text, Valid_label, word_2_index,max_len)
    Valid_loader = DataLoader(Valid_dataset, batch_size, shuffle=False)
 
    model = TextCNNModel(words_embedding, max_len, class_num, channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.load_state_dict(torch.load("./data/model.pth"))
    for e in range(epoch):
        # for batch_idx, batch_label in train_loader:
        #     batch_idx = batch_idx.to(device)
        #     batch_label = batch_label.to(device)
        #     loss = model.forward(batch_idx, batch_label)
        #     loss.backward()
        #     opt.step()
        #     opt.zero_grad()
 
        # print(f"loss:{loss:.3f}")
        model.eval()
        right_num = 0
        for batch_idx, batch_label in Valid_loader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            pre = model.forward(batch_idx)

            right_num += int(torch.sum(pre == batch_label))
        torch.save(model.state_dict(),"./data/model.pth")
        print(f"acc = {right_num/len(Valid_text)*100:.2f}%")



