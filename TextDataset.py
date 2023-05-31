from config import *
import torch
from torch.utils.data import Dataset
classes={'diplomatic_news': 0, 
         'emergency_news': 1, 
         'music_news': 2, 
         'agricultural_news': 3, 
         'stock_news': 4, 
         'game_news': 5, 
         'economic_news': 6, 
         'football_yc': 7, 
         'scientific_news': 8}
class TextDataset(Dataset):
    def __init__(self,text,label,word_2_index,max_len):
        self.text=text
        self.label=label
        self.word_2_index=word_2_index
        self.max_len=max_len
    
    def __getitem__(self, index):
        text = self.text[index][:self.max_len]
        
        label = classes[self.label[index]]
        text_idx = [self.word_2_index.get(i, 1) for i in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))
        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)

        return text_idx, label
    
    def __len__(self):
        return len(self.text)