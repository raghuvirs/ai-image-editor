import streamlit as st
from PIL import Image

# Title of the app
st.title("AI Image Generator - Interactive Photo Editor")

# File uploader to allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# If a file is uploaded, display the image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Image uploaded successfully!")
