import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

COLOR_MAP = {
    (255, 255, 0): 1, # Paratext
    (0, 255, 255): 2, # Decoration
    (255, 0, 255): 3, # Main Text
    (255, 0, 0): 4,   # Title
    (0, 255, 0): 5    # Chapter Headings
}

def rgb_mask_to_class_mask(mask_rgb):
    h, w = mask_rgb.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for color, class_id in COLOR_MAP.items():
        lower = np.array(color, dtype=np.uint8)
        upper = np.array(color, dtype=np.uint8)
        mask_bool = cv2.inRange(mask_rgb, lower, upper) == 255
        class_mask[mask_bool] = class_id
    return class_mask

def main():
    IMG_SRC = "./data/raw/training/images"
    MASK_SRC = "./data/raw/training/masks"
    IMG_OUT = "./data/unet_data/images"
    MASK_OUT = "./data/unet_data/masks"

    os.makedirs(IMG_OUT, exist_ok=True)
    os.makedirs(MASK_OUT, exist_ok=True)

    TILE_SIZE = 512
    AUGMENTATIONS_PER_IMAGE = 30

    transform = A.Compose([
        A.RandomCrop(width=TILE_SIZE, height=TILE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GridDistortion(p=0.3)
    ])

    images = [f for f in os.listdir(IMG_SRC) if f.endswith('.jpg')]
    tile_counter = 0

    for img_name in images:
        img_path = os.path.join(IMG_SRC, img_name)
        mask_path = os.path.join(MASK_SRC, img_name.replace('.jpg', '.png'))
        if not os.path.exists(mask_path): continue

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask_class = rgb_mask_to_class_mask(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB))

        for _ in tqdm(range(AUGMENTATIONS_PER_IMAGE), desc=f"Forging {img_name}"):
            for _ in range(10):
                augmented = transform(image=image, mask=mask_class)
                if np.any(augmented['mask'] > 0):
                    cv2.imwrite(os.path.join(IMG_OUT, f"tile_{tile_counter}.jpg"), cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(MASK_OUT, f"tile_{tile_counter}.png"), augmented['mask'])
                    tile_counter += 1
                    break

    print(f"Forge Complete. Generated {tile_counter} training tiles.")

if __name__ == "__main__":
    main()