import os
import torch
import wandb
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from src.engine.model import TransformerNet, Vgg16, Normalization
from src.engine.dataloader import get_dataloaders
from src.utils.helpers import load_config, gram_matrix, deprocess_image
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def total_variation_loss(image):
    """
    Pikseller arasındaki ani değişimleri cezalandırarak görüntüyü pürüzsüzleştirir.
    Denklem: sum|y_{i,j+1} - y_{i,j}| + sum|y_{i+1,j} - y_{i,j}|
    """
    loss = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def train():
    # 1. Başlangıç ve Yapılandırma
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    
    # W&B Başlatma
    wandb.init(project="neural-style-transfer", config=config)
    
    # 2. Modellerin Hazırlanması
    transformer = TransformerNet().to(device)
    vgg = Vgg16(requires_grad=False).to(device)
    normalization = Normalization(config['vgg_stats']['mean'], config['vgg_stats']['std']).to(device)
    
    optimizer = Adam(transformer.parameters(), lr=float(config['training']['learning_rate']))
    mse_loss = torch.nn.MSELoss()
    
    # 3. Veri Yükleyici
    train_loader = get_dataloaders(config)
    logger.info(f"Eğitim başlıyor! Cihaz: {device}")
    
    # 4. Eğitim Döngüsü
    for epoch in range(int(config['training']['epochs'])):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_tv_loss = 0.
        
        for batch_id, (content_images, style_images) in enumerate(tqdm(train_loader)):
            n_batch = len(content_images)
            optimizer.zero_grad()
            
            content_images = content_images.to(device)
            style_images = style_images.to(device)
            
            # Ressam (TransformerNet) İş Başında
            transformed_images = transformer(content_images)
            
            # VGG için Normalizasyon
            y = normalization(transformed_images)
            xc = normalization(content_images)
            xs = normalization(style_images)
            
            # Özellik Çıkarımı
            features_y = vgg(y)
            features_xc = vgg(xc)
            features_xs = vgg(xs)
            
            # 5. Kayıp (Loss) Hesaplamaları
            c_weight = float(config['loss_weights']['content'])
            s_weight = float(config['loss_weights']['style'])
            tv_weight = float(config['loss_weights']['tv'])
            
            # İçerik Kaybı (Content Loss)
            content_loss = c_weight * mse_loss(features_y.relu2_2, features_xc.relu2_2)
            
            # Stil Kaybı (Style Loss)
            style_loss = 0.
            gram_style = [gram_matrix(y_s) for y_s in features_xs]
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= s_weight
            
            # Pürüzsüzleştirme Kaybı (Total Variation Loss)
            tv_loss = tv_weight * total_variation_loss(transformed_images)
            
            # Toplam Kayıp ve Geri Yayılım
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()
            
            # Metrikleri Biriktir
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_tv_loss += tv_loss.item()
            
            # W&B Loglama ve Görselleştirme
            if (batch_id + 1) % int(config['training']['log_interval']) == 0:
                avg_total = (agg_content_loss + agg_style_loss + agg_tv_loss) / (batch_id + 1)
                
                # Tensor'u deprocess edip görselleştiriyoruz
                sample_img = deprocess_image(transformed_images[0])
                
                wandb.log({
                    "iter": batch_id + (epoch * len(train_loader)),
                    "total_loss": avg_total,
                    "content_loss": agg_content_loss / (batch_id + 1),
                    "style_loss": agg_style_loss / (batch_id + 1),
                    "tv_loss": agg_tv_loss / (batch_id + 1),
                    "stylized_preview": [wandb.Image(sample_img, caption=f"Epoch {epoch} Preview")]
                })

        # Her Epoch sonunda kayıt
        save_path = f"models/transformer_epoch_{epoch}.pth"
        torch.save(transformer.state_dict(), save_path)
        logger.info(f"✅ Epoch {epoch} bitti. Model kaydedildi: {save_path}")

if __name__ == "__main__":
    train()