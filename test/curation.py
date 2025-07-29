import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import tempfile

# --- 1. Config ---
NUM_PALETTE_COLORS = 5
NUM_IMAGES_TO_SELECT = 9

st.set_page_config(page_title="Color Palette Matcher", layout="wide")
st.title("🎨 Color Palette Matcher for Artistic Photography")

# --- 2. Helper: Extract Dominant Colors from Image ---
def extract_dominant_colors(image, k=NUM_PALETTE_COLORS):
    image = image.resize((100, 100))  # speed up
    img_np = np.array(image)
    img_np = img_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(img_np)
    return kmeans.cluster_centers_.astype(int)  # shape: (k, 3)

# --- 3. Helper: Compute Palette Similarity ---
def palette_distance(p1, p2):
    # Flatten and use Euclidean distance
    return np.linalg.norm(p1.flatten() - p2.flatten())

# --- 4. Upload images ---
uploaded_files = st.file_uploader("Sube tus imágenes", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Se subieron {len(uploaded_files)} imágenes. Analizando...")

    # --- 5. Save to temp dir + Extract color palette ---
    image_data = []
    for uploaded in uploaded_files:
        img = Image.open(uploaded).convert("RGB")
        palette = extract_dominant_colors(img)
        image_data.append({"filename": uploaded.name, "image": img, "palette": palette})

    # --- 6. Compute palette similarity matrix ---
    reference_palette = image_data[0]["palette"]
    for item in image_data:
        item["distance"] = palette_distance(reference_palette, item["palette"])

    # --- 7. Select 9 most similar images ---
    sorted_images = sorted(image_data, key=lambda x: x["distance"])
    selected_images = sorted_images[:NUM_IMAGES_TO_SELECT]

    # --- 8. Display as 3x3 grid ---
    st.markdown("### 🎯 Imágenes con paletas similares")
    cols = st.columns(3)
    for i, img_data in enumerate(selected_images):
        with cols[i % 3]:
            st.image(img_data["image"], caption=img_data["filename"], use_container_width="always")
else:
    st.info("📤 Sube al menos una imagen para comenzar.")
