# Neural Style Transfer API Development (14-Day Challenge)

This repository contains an end-to-end, production-ready Neural Style Transfer (NST) system. The goal is to transform user-uploaded images into artistic masterpieces using deep learning.

## 🚀 Project Overview
As part of my journey to become a Global ML Engineer, I am building this project with a focus on **Model Optimization**, **Clean Code**, and **Scalable Deployment**.

### 🛠 Tech Stack
- **Deep Learning:** PyTorch
- **API:** FastAPI
- **Tracking:** Weights & Biases (W&B)
- **Containerization:** Docker
- **Datasets:** COCO (Content) & WikiArt (Style)

## 📈 Roadmap & Progress

- [x] **Day 1: Project Skeleton & Logger Setup**
  - Established a modular folder structure for production.
  - Set up a professional logging system and environment dependencies.

- [x] **Day 2: Data Pipeline & Configuration Management**
  - Implemented `config.yaml` to manage hyperparameters and paths centrally.
  - Created an automated download script for the COCO 2017 dataset.
  - Developed a custom `StyleTransferDataset` to pair content (COCO) and style (WikiArt) images.
  - Built a robust `DataLoader` with preprocessing (resizing, center cropping, and normalization).

- [ ] **Day 3: Model Architecture (Transform Net & Loss Network)**
  - Building the Image Transformation Network and integrating VGG16 for perceptual loss.

---

## 🛠 Setup & Installation
1. Clone the repo: `git clone https://github.com/Batuhanuluer/NeuralStyle-Transfer-API`
2. Install dependencies: `pip install -r requirements.txt`
3. Download data: `python -m src.utils.download_data`