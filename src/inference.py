import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from src.engine.model import TransformerNet
from src.utils.helpers import deprocess_image

def run_inference(model_path, input_image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Modeli Hazırla
    model = TransformerNet().to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval() # Çok kritik! Dropout veya BatchNorm katmanlarını test moduna alır.

    # 2. Resmi Yükle ve Ön İşle (Preprocessing)
    content_image = Image.open(input_image_path).convert('RGB')
    
    # Model 256x256 eğitildiği için şimdilik öyle test edelim
    content_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        # VGG için normalize etmiyoruz çünkü modelin kendi içinde bu beklenmiyor olabilir
        # Ama eğitimde Normalization katmanı eklemiştik, TransformerNet ham resim alıyor.
    ])
    
    content_image = content_transform(content_image).unsqueeze(0).to(device)

    # 3. Modeli Çalıştır (Sanat Başlasın)
    with torch.no_grad():
        output = model(content_image)

    # 4. Resmi Kaydet (Deprocessing)
    # Eğitimdeki deprocess_image fonksiyonumuzu hatırlıyorsun
    output_image = deprocess_image(output[0]) # [0] ile batch boyutundan kurtuluyoruz
    
    # Tensor -> PIL Image
    final_img = transforms.ToPILImage()(output_image)
    final_img.save(output_path)
    print(f"✅ Sanat eseri kaydedildi: {output_path}")

if __name__ == "__main__":
    # Test için yolları ayarla (Burayı kendine göre güncelle)
    MODEL_FILE = "models/transformer_epoch_5.pth"
    INPUT_IMG = "<PATH_TO_YOUR_TEST_IMAGE>" # Buraya bilgisayarındaki bir resmin adını yaz
    OUTPUT_IMG = "output_stylized.jpg"
    
    if os.path.exists(INPUT_IMG):
        run_inference(MODEL_FILE, INPUT_IMG, OUTPUT_IMG)
    else:
        print(f"❌ Hata: {INPUT_IMG} bulunamadı. Lütfen ana dizine bir test resmi koy.")