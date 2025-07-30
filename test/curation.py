import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import tempfile

# Optional: reduce image resolution
def resize_image(img, max_size=(256, 256)):
    return img.resize(max_size, Image.ANTIALIAS)

# Extract average color
def average_color(img):
    arr = np.array(img)
    if arr.ndim == 3:
        return arr.mean(axis=(0, 1))
    elif arr.ndim == 2:  # grayscale
        return np.stack([arr.mean()] * 3)
    else:
        return np.zeros(3)

# Optimize 3x3 grid layout based on visual similarity (placeholder logic)
def optimize_grid(images):
    # Very basic: sort by average brightness
    scored = [(img, average_color(img).mean()) for img in images]
    scored.sort(key=lambda x: x[1])
    return [img for img, _ in scored[:9]]

# Streamlit App
st.set_page_config(layout="wide")
st.title("9-Image Grid Layout Optimizer")

uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing images..."):
        try:
            images = [Image.open(f).convert("RGB") for f in uploaded_files]
            images = [resize_image(img) for img in images]
        except Exception as e:
            st.error(f"Failed to process images: {e}")
            st.stop()

        if len(images) < 9:
            st.warning("Upload at least 9 images for grid optimization.")
        else:
            if st.checkbox("Optimize Grid by Visual Similarity", value=True):
                images = optimize_grid(images)

            # Display grid
            st.subheader("Optimized 3x3 Grid")
            cols = st.columns(3)
            for i in range(9):
                with cols[i % 3]:
                    st.image(images[i], use_column_width=True)
