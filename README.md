# Few-Shot Manuscript Segmentation (U-Net)

A computer vision pipeline for semantic layout analysis of historical manuscripts, trained under severe data scarcity (few-shot learning). 

This repository demonstrates how to effectively train a ResNet34-backed U-Net architecture using only 3 manually annotated ground-truth images by utilizing heavy synthetic data generation and sliding-window inference.

## Architecture & Techniques

* **Synthetic Data Forging:** Utilizes `albumentations` to generate 3,000 highly augmented training tiles (grid distortions, rotations, color shifts) from 3 base images, solving the data scarcity problem.
* **Semantic Segmentation:** Implements a U-Net architecture with a pre-trained ResNet34 backbone (`segmentation-models-pytorch`) to classify 5 distinct manuscript elements (Main Text, Titles, Decorations, Paratext, Chapter Headings).
* **Sliding Window Inference:** Processes massive, high-resolution test images (4K+) by utilizing a strided sliding window algorithm, avoiding the destructive downscaling typically required by memory constraints.
* **Web UI:** Includes a standalone `gradio` web application for interactive model demonstration.

## Quick Start

### Installation
```bash
git clone [https://github.com/YOUR_USERNAME/few-shot-segmentation.git](https://github.com/YOUR_USERNAME/few-shot-segmentation.git)
cd few-shot-segmentation
pip install -r requirements.txt