import gc
import os
import os.path as osp
import cv2
import sys
import json
import tqdm
import time
import timm
import torch
import random
import sklearn.metrics

from PIL import Image
from pathlib import Path
from functools import partial
from contextlib import contextmanager

import numpy as np
import scipy as sp
import pandas as pd
import torch.nn as nn

from torch.optim import Adam, SGD
from scipy.special import softmax
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


class TestDataset(Dataset):
    def __init__(self, df, selected_features: list, transform=None):
        self.df = df
        self.selected_features = selected_features
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_path = self.df['image_path'].values[idx]
        label = self.df['class_id'].values[idx]  
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        selected_feature_values = {feature: self.df[feature].values[idx] for feature in self.selected_features}
            
        return image, label, file_path, selected_feature_values


def get_transforms(model_mean, model_std, image_size):

    return Compose([Resize(*image_size),
                    Normalize(mean=model_mean, std=model_std),
                    ToTensorV2()])
