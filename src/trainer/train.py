import os
import torch
import wandb
from tqdm import tqdm
from torch.optim import Adam
from src.engine.model import TransformerNet, Vgg16, Normalization
from src.engine.dataloader import get_dataloaders
from src.utils.helpers import load_config, gram_matrix
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def train():
    # 1. Setup & Config
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Checkpoints klasörünü oluştur
    os.makedirs("models", exist_ok=True)
    
    # Initialize W&B tracking
    wandb.init(project="neural-style-transfer", config=config)
    
    # 2. Initialize Models
    transformer = TransformerNet().to(device)
    vgg = Vgg16(requires_grad=False).to(device)
    normalization = Normalization(config['vgg_stats']['mean'], config['vgg_stats']['std']).to(device)
    
    optimizer = Adam(transformer.parameters(), lr=float(config['training']['learning_rate']))
    mse_loss = torch.nn.MSELoss()
    
    # 3. Data
    train_loader = get_dataloaders(config)
    
    logger.info(f"Starting training session on {device}...")
    
    # 4. Training Loop
    for epoch in range(int(config['training']['epochs'])):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        
        for batch_id, (content_images, style_images) in enumerate(tqdm(train_loader)):
            n_batch = len(content_images)
            optimizer.zero_grad()
            
            content_images = content_images.to(device)
            style_images = style_images.to(device)
            
            # Generate Stylized Image
            transformed_images = transformer(content_images)
            
            # Normalize for VGG (ImageNet stats)
            y = normalization(transformed_images)
            xc = normalization(content_images)
            xs = normalization(style_images)
            
            # Extract Features
            features_y = vgg(y)
            features_xc = vgg(xc)
            features_xs = vgg(xs)
            
            # 5. Calculate Losses (Explicit float casting to prevent index errors)
            c_weight = float(config['loss_weights']['content'])
            s_weight = float(config['loss_weights']['style'])
            
            # Content Loss (relu2_2)
            content_loss = c_weight * mse_loss(features_y.relu2_2, features_xc.relu2_2)
            
            # Style Loss (Gram Matrix on all 4 layers)
            style_loss = 0.
            # Gram matrices for style reference images
            gram_style = [gram_matrix(y_s) for y_s in features_xs]
            
            # Iterate through the namedtuple outputs
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            
            style_loss *= s_weight
            
            # Total Loss & Backprop
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            
            # Log metrics
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            
            # train.py içindeki log kısmını bul ve şununla değiştir:
            if (batch_id + 1) % int(config['training']['log_interval']) == 0:
                from src.utils.helpers import deprocess_image # Importu ekle
                
                avg_total = (agg_content_loss + agg_style_loss) / (batch_id + 1)
                
                # Resmi insan gözüne uygun hale getir
                sample_img = deprocess_image(transformed_images[0])
                
                wandb.log({
                    "batch": batch_id + (epoch * len(train_loader)),
                    "total_loss": avg_total,
                    "stylized_preview": [wandb.Image(sample_img, caption=f"Epoch {epoch} Preview")]
                })

        # Save Checkpoint
        save_path = f"models/transformer_epoch_{epoch}.pth"
        torch.save(transformer.state_dict(), save_path)
        logger.info(f"✅ Epoch {epoch} completed. Checkpoint saved: {save_path}")

if __name__ == "__main__":
    train()