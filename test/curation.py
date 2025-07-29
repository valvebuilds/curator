import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import io
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


st.set_page_config(layout="wide")
st.title("🎨 Artistic Photo Color Palette Matcher")

# --- Helper Functions ---

def extract_dominant_colors(image, k=5):
    image = image.resize((100, 100))  # Downsample for speed
    img_array = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(img_array)
    counts = np.bincount(labels)
    centers = kmeans.cluster_centers_.astype(int)
    sorted_colors = centers[np.argsort(-counts)]
    return sorted_colors  # (k, 3)

def rgb_to_lab(color):
    rgb = sRGBColor(*color, is_upscaled=True)
    lab = convert_color(rgb, LabColor)
    return lab

def compute_palette_distance(palette1, palette2):
    lab1 = [rgb_to_lab(c) for c in palette1]
    lab2 = [rgb_to_lab(c) for c in palette2]
    dists = [min([delta_e_cie2000(c1, c2) for c2 in lab2]) for c1 in lab1]
    return np.mean(dists)

# --- File Upload ---
uploaded_files = st.file_uploader("Sube tus imágenes (30+ JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 9:
    with st.spinner("Procesando imágenes..."):

        images = []
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            palette = extract_dominant_colors(img, k=5)
            images.append({
                "image": img,
                "palette": palette,
                "filename": file.name
            })

        # Compute average palette of all images
        all_palettes = np.array([img["palette"].mean(axis=0) for img in images])
        mean_palette = all_palettes.mean(axis=0)

        # Compute distance to mean palette
        for img in images:
            img["distance"] = compute_palette_distance(img["palette"], [mean_palette])

        # Sort by similarity
        images.sort(key=lambda x: x["distance"])
        top_images = images[:9]

        st.success("✅ ¡Listo! Mostrando el grid de imágenes más armoniosas por color:")

        # Display 3x3 grid
        for i in range(0, 9, 3):
            cols = st.columns(3)
            for j in range(3):
                with cols[j]:
                    img_data = top_images[i + j]
                    st.image(img_data["image"], caption=img_data["filename"], use_column_width=True)

else:
    st.warning("Por favor sube al menos 9 imágenes para generar el grid.")
