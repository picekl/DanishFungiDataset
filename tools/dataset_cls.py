import cv2
import pandas as pd
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class ExtraFeaturesDataset(Dataset):
    """ Supply extra features from the dataset."""
    def __init__(
            self,
            df: pd.DataFrame,
            image_path_feature: str,
            target_feature: str,
            extra_features: list[str],
            transform=None
    ):
        self.df = df
        self.image_path_feature = image_path_feature
        self.target_feature = target_feature
        self.extra_features = extra_features
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def get_extra_features_names(self):
        return self.extra_features

    def __getitem__(self, idx: int) -> tuple:
        file_path = self.df[self.image_path_feature].values[idx]
        label = self.df[self.target_feature].values[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        selected_feature_values = {
            feature: self.df[feature].values[idx] for feature in self.extra_features
        }

        return image, label, file_path, selected_feature_values


def get_transforms(model_mean, model_std, image_size):
    return Compose(
        [Resize(*image_size), Normalize(mean=model_mean, std=model_std), ToTensorV2()]
    )
