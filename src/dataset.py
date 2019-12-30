import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        logmelspec = self.dataset[index]['features']['logmelspec']
        mfcc = self.dataset[index]['features']['mfcc']
        chroma = self.dataset[index]['features']['chroma']
        spec_contrast = self.dataset[index]['features']['spectral_contrast']
        tonnetz = self.dataset[index]['features']['tonnetz']

        if self.mode == 'LMC':
            feature = np.concatenate((logmelspec, chroma, spec_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            feature = np.concatenate((mfcc, chroma, spec_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            feature = np.concatenate((mfcc, logmelspec, chroma, spec_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
       
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)
