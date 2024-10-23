# AI Image Editor - Interactive Photo Editor

## Overview
This project is an AI-powered interactive photo editor that allows users to apply various effects on images using object detection. Users can upload images and apply modifications such as blurring, sharpening, edge detection, and more to specific objects within the image.

## Features
- Object detection using a pre-trained DETR model.
- Various image modifications, including:
  - Adjust brightness
  - Adjust contrast
  - Adjust saturation
  - Grayscale conversion
  - Edge detection
  - Sepia filter
  - Pixelation
  - Color inversion
  - Image rotation
- Interactive user interface built with Streamlit.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/raghuvirs/ai-image-editor.git
   cd <repository_directory>
2. Install required package:
    ```bash
    pip install -r requirements.txt
3. Run the Streamlit application:
    ```bash
    streamlit run app.py