import glob
import os
from PIL import Image
from torch.utils.data import Dataset

class StyleTransferDataset(Dataset):
    """
    Robust Dataset for Neural Style Transfer. 
    Handles nested directories for style images (WikiArt).
    """
    def __init__(self, content_dir, style_dir, transform=None):
        # Recursively find all images in nested subfolders
        self.content_images = glob.glob(os.path.join(content_dir, "**/*.jpg"), recursive=True)
        self.style_images = glob.glob(os.path.join(style_dir, "**/*.jpg"), recursive=True)
        self.transform = transform

        if len(self.content_images) == 0 or len(self.style_images) == 0:
            raise RuntimeError(f"Found 0 images in {content_dir} or {style_dir}. Check paths!")

    def __len__(self):
        return len(self.content_images)

    def __getitem__(self, idx):
        content_img = Image.open(self.content_images[idx]).convert('RGB')
        
        # Randomly sample style image for more variety during training
        import random
        style_path = random.choice(self.style_images)
        style_img = Image.open(style_path).convert('RGB')

        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)

        return content_img, style_img