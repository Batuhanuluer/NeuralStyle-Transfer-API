from torch.utils.data import DataLoader
from torchvision import transforms
from src.engine.dataset import StyleTransferDataset

def get_dataloaders(config):
    """
    Create and return the DataLoader for training.
    """
    # Standard NST transformations: Resize, Convert to Tensor, and Normalize
    transform = transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.CenterCrop(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = StyleTransferDataset(
        content_dir=config['data']['content_dir'],
        style_dir=config['data']['style_dir'],
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )

    return loader