from torch.utils.data import Dataset
from PIL import Image
import torch
import os


class CelebaFaceDataset(Dataset):
    def __init__(self, img_dir, attr_df, partition_df, split=0, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        partition_imgs = set(partition_df[partition_df["partition"] == split]["image_id"].values)
        self.attr_df = attr_df[attr_df["image_id"].isin(partition_imgs)].reset_index(drop=True)

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        row = self.attr_df.iloc[idx]
        img_name = row["image_id"]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        gender = 1.0 if row["Male"] == 1 else 0.0
        return img, torch.tensor(gender, dtype=torch.float32)