import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class UNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, img_name)), cv2.COLOR_BGR2RGB) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = cv2.imread(os.path.join(self.mask_dir, img_name.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

def main():
    IMG_DIR = "./data/unet_data/images"
    MASK_DIR = "./data/unet_data/masks"
    SAVE_DIR = "./weights"
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = DataLoader(UNetDataset(IMG_DIR, MASK_DIR), batch_size=8, shuffle=True)
    
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"unet_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()