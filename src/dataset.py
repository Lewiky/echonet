import torch
from torch.utils import data
import numpy as np
import math
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode, augmentation_length: int = 1):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode
        #augmentation length = number of times to go through dataset with augmentations
        self.augmentation_length = augmentation_length

    def _time_shift(self, tensor: torch.Tensor, amount: int) -> torch.Tensor:
        '''
        Given some `tensor`, take the first `amount` cols from the front of the spectogram 
        and move them to the end, effectively 'shifting' the sound
        '''
        return torch.cat([tensor[:,amount:], tensor[:,:amount]], dim=1)

    def __getitem__(self, index):
        cut_percentage = None
        offset = None
        dataset_length = len(self.dataset)

        #If we're going around the dataset again, e.g into augmented data
        if index > dataset_length:
            cut_percentage = (index - dataset_length) / dataset_length
            offset = 1/(index // dataset_length)
            index = index % dataset_length

        logmelspec = self.dataset[index]['features']['logmelspec']
        mfcc = self.dataset[index]['features']['mfcc']
        chroma = self.dataset[index]['features']['chroma']
        spec_contrast = self.dataset[index]['features']['spectral_contrast']
        tonnetz = self.dataset[index]['features']['tonnetz']

        if cut_percentage is not None:
            shift_amount = math.floor(len(logmelspec[0]) * (offset + cut_percentage)) % dataset_length
            logmelspec = self._time_shift(logmelspec, shift_amount)
            mfcc = self._time_shift(mfcc, shift_amount)
            chroma = self._time_shift(chroma, shift_amount)
            spec_contrast = self._time_shift(spec_contrast, shift_amount)
            tonnetz = self._time_shift(tonnetz, shift_amount)

        if self.mode == 'LMC':
            feature = np.concatenate((logmelspec, chroma, spec_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            feature = np.concatenate((mfcc, chroma, spec_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            feature = np.concatenate((mfcc, logmelspec, chroma, spec_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == "TSCNN":
            feature_lmc = np.concatenate((logmelspec, chroma, spec_contrast, tonnetz), axis=0)
            feature_lmc = torch.from_numpy(feature_lmc.astype(np.float32)).unsqueeze(0)
            feature_mc = np.concatenate((mfcc, chroma, spec_contrast, tonnetz), axis=0)
            feature_mc = torch.from_numpy(feature_mc.astype(np.float32)).unsqueeze(0)
            feature = (feature_lmc, feature_mc)
       
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname, index

    def __len__(self):
        return self.augmentation_length * len(self.dataset)
