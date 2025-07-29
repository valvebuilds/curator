import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from colorspacious import cspace_convert
from PIL import Image
import itertools
import math
import random
import os
import tempfile

# ---------- UTILITIES ---------- #

def rgb_to_hsv(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0] / [180, 255, 255]

def rgb_to_lab(rgb):
    rgb_arr = np.uint8([[rgb]])
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
    return lab[0][0]

def hue_distance(h1, h2):
    return min(abs(h1 - h2), 1 - abs(h1 - h2)) * 360

def color_temperature(rgb):
    r, g, b = rgb
    return 'warm' if r > b else 'cool'

# ---------- COLOR EXTRACTION ---------- #

def extract_color_palette(img, n_colors=5):
    image_np = np.array(img.convert('RGB').resize((100, 100)))
    img_data = image_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(img_data)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    hsv_colors = [rgb_to_hsv(tuple(rgb)) for rgb in dominant_colors]
    lab_colors = [rgb_to_lab(tuple(rgb)) for rgb in dominant_colors]

    return {
        'rgb': [tuple(c) for c in dominant_colors],
        'hsv': hsv_colors,
        'lab': lab_colors
    }

# ---------- SCORING ---------- #

def calculate_harmony_score(palettes):
    hue_diffs = []
    sat_diffs = []
    temperature_balance = 0

    for (p1, p2) in itertools.combinations(palettes, 2):
        for c1 in p1['hsv']:
            for c2 in p2['hsv']:
                hue_diffs.append(hue_distance(c1[0], c2[0]))
                sat_diffs.append(abs(c1[1] - c2[1]))

    for p in palettes:
        for rgb in p['rgb']:
            temperature_balance += 1 if color_temperature(rgb) == 'warm' else -1

    mean_hue_diff = np.mean(hue_diffs)
    mean_sat_diff = np.mean(sat_diffs)
    temperature_score = 1.0 - abs(temperature_balance) / (len(palettes) * 5)

    harmony_score = 1.0 / (1 + mean_hue_diff + mean_sat_diff) + temperature_score
    return round(harmony_score, 4)

def calculate_grid_harmony(images):
    palettes = [extract_color_palette(img) for img in images]
    return calculate_harmony_score(palettes)

# ---------- OPTIMIZATION ---------- #

def score_grid(images):
    return calculate_grid_harmony(images)

def optimize_grid(images, iterations=100):
    best_layout = images[:]
    best_score = score_grid(images)

    for _ in range(iterations):
        shuffled = images[:]
        random.shuffle(shuffled)
        score = score_grid(shuffled)
        if score > best_score:
            best_score = score
            best_layout = shuffled

    return best_layout, best_score

def create_grid_image(images, grid_size=(3,3), image_size=(200, 200)):
    rows, cols = grid_size
    grid_img = Image.new('RGB', (cols * image_size[0], rows * image_size[1]))

    for idx, img in enumerate(images):
        resized = img.resize(image_size)
        row, col = divmod(idx, cols)
        grid_img.paste(resized, (col * image_size[0], row * image_size[1]))

    return grid_img

# ---------- STREAMLIT APP ---------- #

st.set_page_config(page_title="Color Grid Validator", layout="wide")
st.title("🎨 Social Media Grid Color Harmony Validator")
st.markdown("Upload 9 images to evaluate and optimize your Instagram-style grid based on color harmony.")

uploaded_files = st.file_uploader("Upload exactly 9 images", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 9:
    st.subheader("Original Grid Preview")
    images = [Image.open(file) for file in uploaded_files]
    original_grid = create_grid_image(images)
    original_score = score_grid(images)
    st.image(original_grid, caption=f"Harmony Score: {original_score}", use_column_width=True)

    st.subheader("Optimizing Layout...")
    best_layout, best_score = optimize_grid(images, iterations=50)
    optimized_grid = create_grid_image(best_layout)
    st.image(optimized_grid, caption=f"Optimized Harmony Score: {best_score}", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        optimized_grid.save(tmp.name)
        st.download_button(
            label="Download Optimized Grid",
            data=open(tmp.name, "rb"),
            file_name="optimized_grid.jpg",
            mime="image/jpeg"
        )

elif uploaded_files:
    st.warning("Please upload exactly **9** images to fill a 3x3 grid.")

else:
    st.info("👆 Upload your image set to get started.")
