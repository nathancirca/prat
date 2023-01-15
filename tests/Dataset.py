import numpy as np
import torch
from utils import *



class Dataset(torch.utils.data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, patche):
        self.patches = patche

        # self.patches_MR = patches_MR
        # self.patches_CT = patches_CT

    def __len__(self):
        'denotes the total number of samples'
        return len(self.patches)

    def __getitem__(self, index):
        'Generates one sample of data'
        #select sample
        batch_clean = (self.patches[index,:, :, 0])
        x = torch.tensor(batch_clean)
        x = x.unsqueeze(0)


        return x


class ValDataset(torch.utils.data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, test_set):
        self.files = glob(test_set+'/*.npy')

        # self.patches_MR = patches_MR
        # self.patches_CT = patches_CT

    def __len__(self):
        'denotes the total number of samples'
        return len(self.files)

    def __getitem__(self, index):
        'Generates one sample of data'
        #select sample
        eval_data = load_sar_images(self.files)
        current_test=normalize_sar(eval_data[index])
        return torch.tensor(current_test).type(torch.float)
