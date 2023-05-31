from config import *
import matplotlib.pyplot as plt
import json
def count_text_len():
    text_len=[]
    with open(Train_sample_path,'r',encoding='utf-8') as f:
        for dict in json.load(f):
            text=dict['text']
            text_len.append(len(text))
    plt.hist(len(text))
    plt.show()
    print(max(text_len))
if __name__=='__main__':
    count_text_len()