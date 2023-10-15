from typing import Any, Tuple, Optional, Callable
import os
import glob

import torch
from torch.utils.data import Dataset
from utils import *


class CIFAR10Dataset(Dataset):
    def __init__(
            self,
            args,
            train: bool,
            transform = None
        ):
        super().__init__()
        
        self.data_dir = args.data_dir
        self.train = train
        self.transform = transform
        
        self.data = None
        self.labels = []
        
        # Load Data
        if train:
            self.file_list = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            self.file_list = ["test_batch"]
        
        data_list = []
        for file_name in self.file_list:
            file_dir = os.path.join(self.data_dir, file_name)
            # Add Data
            data = torch.from_numpy(unpickle(file_dir)[b'data'].reshape(-1, 3, 32, 32))
            data_list.append(data)
            # Add Labels
            label = unpickle(file_dir)[b'labels']
            self.labels.extend(label)
        
        # Concatenate Data
        self.data = torch.cat(data_list, dim=0)
        self.data = self.data / 255   # normalize to range 0~1
        
    def __getitem__(self, idx: int):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data, label
    
    def __len__(self):
        return len(self.labels)