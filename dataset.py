import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
class SegDataset(Dataset):
    def __init__(self, files):
        self.data = []
        for i in files:
            with open('data/{}'.format(i), 'rb') as f:
                d = pickle.load(f)
            self.data.append(d)
            
    def __len__(self):
        return len(self.data)
    
    
    
    def __getitem__(self, index):
        
        place = self.data[index]['place']
        cast = self.data[index]['cast']
        action = self.data[index]['action']
        audio = self.data[index]['audio']
        
        output = self.data[index]['scene_transition_boundary_ground_truth']
        input_feat = torch.cat((place,cast,action,audio),1)
        
        coarse = self.data[index]['scene_transition_boundary_prediction'].numpy()
        coarse = coarse >= 0.5
        coarse = coarse.astype(int)
        coarse = np.append(coarse,[1])
        coarse = torch.LongTensor(coarse)
        output = output.float()
        mask = torch.ones(place.shape[0],dtype=torch.long)
        global_mask = coarse.clone().detach()
        
        ids = self.data[index]['imdb_id']
        
        
        return input_feat, coarse, mask, global_mask, output, ids