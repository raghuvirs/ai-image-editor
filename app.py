import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection, pipeline
import torch
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
from skimage.restoration import inpaint

# Title of the app
st.title("AI Image Generator - Interactive Photo Editor")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Text input for user prompt
user_prompt = st.text_input("Enter a command (e.g., 'Blur the person in the background'):")

# Specify a zero-shot classification model explicitly for the NLP pipeline
nlp_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible actions and objects
actions = ["blur", "sharpen", "remove"]
objects = ["person", "background", "kite", "bucket"]

if uploaded_file is not None and user_prompt:
    # Run NLP model to classify action and object from the prompt
    action_results = nlp_pipeline(user_prompt, actions)
    object_results = nlp_pipeline(user_prompt, objects)
    
    # Extract action and object
    selected_action = action_results["labels"][0]  # most likely action
    selected_object = object_results["labels"][0]  # most likely object

    st.write(f"Detected Action: {selected_action}")
    st.write(f"Detected Object: {selected_object}")

    # Load the image
    image = Image.open(uploaded_file)

    # Load the Detr model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Preprocess the image for the Detr model
    inputs = processor(images=image, return_tensors="pt")

    # Perform object detection
    outputs = model(**inputs)

    # Let's only keep detections with a high score
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Draw bounding boxes and modify image based on the selected action
    np_image = np.array(image)  # Convert PIL image to NumPy array for OpenCV processing
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        if model.config.id2label[label.item()] == selected_object:
            # Perform action: blur, remove, or sharpen
            if selected_action == "blur":
                # Extract the region to blur
                x1, y1, x2, y2 = map(int, box)
                object_region = np_image[y1:y2, x1:x2]
                blurred_region = cv2.GaussianBlur(object_region, (15, 15), 0)
                np_image[y1:y2, x1:x2] = blurred_region
            elif selected_action == "remove":
                # Inpainting to remove object
                x1, y1, x2, y2 = map(int, box)
                mask = np.zeros(np_image.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                np_image = inpaint.inpaint_biharmonic(np_image, mask, multichannel=True)
            elif selected_action == "sharpen":
                # Sharpen the region
                x1, y1, x2, y2 = map(int, box)
                object_region = np_image[y1:y2, x1:x2]
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
                sharpened_region = cv2.filter2D(object_region, -1, kernel)
                np_image[y1:y2, x1:x2] = sharpened_region

    # Convert NumPy array back to PIL image
    modified_image = Image.fromarray(np.uint8(np_image))

    # Display the image with modifications
    st.image(modified_image, caption="Processed Image with Modifications", use_column_width=True)

