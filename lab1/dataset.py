from torch.utils.data import DataLoader, Dataset
from PIL import Image


class CarColorDataset(Dataset):
    def __init__(self, image_paths, labels, color_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.color_to_idx = color_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.color_to_idx[self.labels[idx]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label