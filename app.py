import gradio as gr
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=6)
model.load_state_dict(torch.load("./weights/unet_epoch_10.pt", map_location=device))
model.to(device)
model.eval()

COLORS = {
    1: [255, 255, 0],   # Paratext
    2: [0, 255, 255],   # Decoration
    3: [255, 0, 255],   # Main Text
    4: [255, 0, 0],     # Title
    5: [0, 255, 0]      # Chapter
}

def predict_manuscript(input_image):
    if input_image is None: return None
    original_h, original_w, _ = input_image.shape
    img_resized = cv2.resize(input_image, (1024, 1024))
    
    img_tensor = torch.tensor(np.transpose(img_resized / 255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = torch.argmax(torch.softmax(model(img_tensor), dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    overlay = np.zeros_like(img_resized)
    for class_id, color in COLORS.items():
        overlay[pred_mask == class_id] = color

    blended = cv2.addWeighted(img_resized, 0.7, overlay, 0.5, 0)
    return cv2.resize(blended, (original_w, original_h))

demo = gr.Interface(
    fn=predict_manuscript,
    inputs=gr.Image(type="numpy", label="Upload Raw Manuscript Page"),
    outputs=gr.Image(type="numpy", label="U-Net AI Prediction"),
    title="Medieval Manuscript Segmenter (U-Net)",
    description="Few-shot semantic segmentation trained on 3 annotated pages via Albumentation forging. Upload a historical manuscript to map the text geometry.",
    theme="huggingface"
)

if __name__ == "__main__":
    demo.launch(share=False)