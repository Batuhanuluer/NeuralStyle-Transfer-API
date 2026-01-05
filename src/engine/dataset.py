import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StyleTransferDataset(Dataset):
    """
    Custom Dataset for Neural Style Transfer.
    It pairs content images from COCO with style images from WikiArt.
    """
    def __init__(self, content_dir, style_dir, transform=None):
        """
        :param content_dir: Path to content images directory.
        :param style_dir: Path to style images directory.
        :param transform: PyTorch transforms to be applied on images.
        """
        self.content_images = [os.path.join(content_dir, i) for i in os.listdir(content_dir) 
                               if i.endswith(('.png', '.jpg', '.jpeg'))]
        self.style_images = [os.path.join(style_dir, i) for i in os.listdir(style_dir)
                             if i.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        # We define the epoch length by the number of content images
        return len(self.content_images)

    def __getitem__(self, idx):
        # Load content image
        content_path = self.content_images[idx]
        content_img = Image.open(content_path).convert('RGB')

        # Select a style image (Cycle through style images if they are fewer than content images)
        style_path = self.style_images[idx % len(self.style_images)]
        style_img = Image.open(style_path).convert('RGB')

        # Apply transformations (Resizing, Normalization, Tensor conversion)
        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)

        return content_img, style_img