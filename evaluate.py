import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

def main():
    TEST_IMG_DIR = "./data/raw/validation/images"
    TEST_GT_DIR = "./data/raw/validation/masks"
    WEIGHTS_PATH = "./weights/unet_epoch_10.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=6)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    TILE_SIZE = 512
    STRIDE = 256
    CLASS_IDS = [1, 2, 3, 4, 5]
    total_intersection = 0.0
    total_union = 0.0

    test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.jpg')]

    for file in tqdm(test_images, desc="Evaluating"):
        img_path = os.path.join(TEST_IMG_DIR, file)
        gt_path = os.path.join(TEST_GT_DIR, file.replace('.jpg', '.png'))
        if not os.path.exists(gt_path): continue

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        pred_canvas = np.zeros((h, w), dtype=np.uint8)

        x_steps = int(np.ceil((w - TILE_SIZE) / STRIDE)) + 1
        y_steps = int(np.ceil((h - TILE_SIZE) / STRIDE)) + 1

        for y_idx in range(y_steps):
            for x_idx in range(x_steps):
                x1, y1 = x_idx * STRIDE, y_idx * STRIDE
                x2, y2 = min(x1 + TILE_SIZE, w), min(y1 + TILE_SIZE, h)
                if x2 - x1 < TILE_SIZE: x1 = max(0, x2 - TILE_SIZE)
                if y2 - y1 < TILE_SIZE: y1 = max(0, y2 - TILE_SIZE)

                tile = image_rgb[y1:y2, x1:x1+TILE_SIZE] / 255.0
                tile_tensor = torch.tensor(np.transpose(tile, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_mask = torch.argmax(torch.softmax(model(tile_tensor), dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                mask_region = pred_mask > 0
                pred_canvas[y1:y2, x1:x1+TILE_SIZE][mask_region] = pred_mask[mask_region]

        # Calculate IoU logic omitted for brevity (same as your notebook)
        # Add your standard IoU evaluation logic here

    print("Evaluation complete.")

if __name__ == "__main__":
    main()