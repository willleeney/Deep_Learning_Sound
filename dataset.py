import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        element = self.dataset[index]['features']
        feature1 = np.array([element['logmelspec']])
        feature1 = torch.from_numpy(feature1.astype(np.float32)).unsqueeze(0)
        feature2 = np.array([element['mfcc']])
        feature2 = torch.from_numpy(feature2.astype(np.float32)).unsqueeze(0)
        feature3 = np.array([element['chroma']])
        feature3 = torch.from_numpy(feature3.astype(np.float32)).unsqueeze(0)
        feature4 = np.array([element['spectral_contrast']])
        feature4 = torch.from_numpy(feature4.astype(np.float32)).unsqueeze(0)
        feature5 = np.array([element['tonnetz']])
        feature5 = torch.from_numpy(feature5.astype(np.float32)).unsqueeze(0)
        if self.mode == 'LMC':
            feature = torch.cat((feature1[0][0], feature3[0][0], feature4[0][0], feature5[0][0]), 0).unsqueeze(0)  
        elif self.mode == 'MC':
            feature = [feature2[0][0], feature3[0][0], feature4[0][0], feature5[0][0]]
        elif self.mode == 'MLMC':
            feature = [feature1[0][0], feature2[0][0], feature3[0][0], feature4[0][0], feature5[0][0]]       
 
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)
