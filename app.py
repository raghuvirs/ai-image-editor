import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection, pipeline
import torch
from PIL import Image
import cv2
import numpy as np
from skimage.restoration import inpaint
from io import BytesIO
from datetime import datetime

# Function Definitions for Image Modifications
def adjust_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    final_hsv = cv2.merge((h, s, v))
    bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return bright_image

def adjust_contrast(image, alpha=1.5):
    contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return contrast_image

def adjust_saturation(image, saturation_scale=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, saturation_scale)
    final_hsv = cv2.merge((h, s, v))
    saturated_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return saturated_image

def grayscale_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

def pixelate(image, pixel_size=10):
    height, width = image.shape[:2]
    small_image = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image

def invert_colors(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

def rotate_image(image, angle=45):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def image_download_button(pil_image, filename: str, fmt: str, label="Download"):
    buf = BytesIO()
    pil_image.save(buf, format=fmt.upper())
    return st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=f"{filename}.{fmt}",
        mime=f"image/{fmt}"
    )

# Title of the app
st.title("AI Image Editor - Interactive Photo Editor")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Load the Detr model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Specify a zero-shot classification model for NLP
nlp_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible actions and objects
actions = [
    "blur", "sharpen", "remove",
    "brightness", "contrast", "saturation", 
    "grayscale", "edge", "sepia", 
    "pixelate", "invert color", "rotate"
]
objects = ["person", "background", "kite", "bucket","man","woman", "girl","boy"]

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # User Input for Image Modification
    user_prompt = st.text_input("Enter your command (e.g., 'Apply edge detection to the person')")

    if st.button("Apply Modification"):
        # Run NLP model to classify action and object from the prompt
        action_results = nlp_pipeline(user_prompt, actions)
        object_results = nlp_pipeline(user_prompt, objects)

        # Extract action and object
        selected_action = action_results["labels"][0]  # Most likely action
        selected_object = object_results["labels"][0]  # Most likely object

        st.write(f"Detected Action: {selected_action}")
        st.write(f"Detected Object: {selected_object}")

        # Convert PIL image to NumPy array for OpenCV processing
        np_image = np.array(image)

        if selected_action in ["blur", "sharpen", "remove", "edge detection"]:
            # Preprocess the image for the Detr model
            inputs = processor(images=image, return_tensors="pt")

            # Perform object detection
            outputs = model(**inputs)

            # Let's only keep detections with a high score
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            detected_boxes = []
            # Draw bounding boxes and modify image based on the selected action
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                detected_label = model.config.id2label[label.item()]
                detected_boxes.append({"label": detected_label, "score": score.item(), "box": box})

                # Check if the detected object matches the selected object
                if detected_label == selected_object:
                    x1, y1, x2, y2 = map(int, box)

                    if selected_action == "edge detection":
                        # Extract the region for edge detection
                        object_region = np_image[y1:y2, x1:x2]
                        edged_region = edge_detection(object_region)
                        np_image[y1:y2, x1:x2] = edged_region
                        st.write("Edge detection applied to the detected object.")
                    
                    elif selected_action == "blur":
                        # Extract the region to blur
                        object_region = np_image[y1:y2, x1:x2]
                        blurred_region = cv2.GaussianBlur(object_region, (35, 35), 0)
                        np_image[y1:y2, x1:x2] = blurred_region
                        st.write("Blurring applied to the detected object.")
                    
                    elif selected_action == "remove":
                        with st.spinner("AI is doing the magic!"):
                            # Create the mask from the bounding box
                            mask = np.zeros(np_image.shape[:2], dtype=np.uint8)  # Mask should be 2D
                            mask[y1:y2, x1:x2] = 1  # Fill in the mask
                            
                            # Inpainting to remove the object
                            output = inpaint.inpaint_biharmonic(np_image, mask, channel_axis=2)  # Inpainting
                            
                            img_output = Image.fromarray((output * 255).astype(np.uint8))  # Convert back to image
                            
                        st.write("AI has finished the job!")
                        st.image(img_output)

                       
                    
                    elif selected_action == "sharpen":
                        # Sharpen the region
                        object_region = np_image[y1:y2, x1:x2]
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
                        sharpened_region = cv2.filter2D(object_region, -1, kernel)
                        np_image[y1:y2, x1:x2] = sharpened_region
                        st.write("Sharpening applied to the detected object.")

        # Handle additional actions
        if selected_action == "adjust brightness":
            np_image = adjust_brightness(np_image)
        elif selected_action == "adjust contrast":
            np_image = adjust_contrast(np_image)
        elif selected_action == "adjust saturation":
            np_image = adjust_saturation(np_image)
        elif selected_action == "grayscale":
            np_image = grayscale_object(np_image)
        elif selected_action == "apply sepia":
            np_image = apply_sepia(np_image)
        elif selected_action == "pixelate":
            np_image = pixelate(np_image)
        elif selected_action == "invert colors":
            np_image = invert_colors(np_image)
        elif selected_action == "rotate":
            np_image = rotate_image(np_image)

        # Convert NumPy array back to PIL image
        modified_image = Image.fromarray(np.uint8(np_image))

        # Display the modified image
        st.image(modified_image, caption="Processed Image with Modifications", use_column_width=True)
