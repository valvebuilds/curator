import os
import warnings
import sys

# ✅ Disable inotify file watchers (fix for Streamlit Cloud)
os.environ["STREAMLIT_WATCHDOG_MODE"] = "poll"

# ✅ Suppress SyntaxWarnings early
if not sys.warnoptions:
    warnings.simplefilter("ignore", SyntaxWarning)
import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from colorspacious import cspace_convert
from PIL import Image
import itertools
import random
import tempfile

# ---------- UTILITIES ---------- #

def rgb_to_hsv(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0] / [180, 255, 255]

def hue_distance(h1, h2):
    return min(abs(h1 - h2), 1 - abs(h1 - h2)) * 360

def color_temperature(rgb):
    r, g, b = rgb
    return 'warm' if r > b else 'cool'

def extract_color_palette(img, n_colors=5):
    image_np = np.array(img.convert('RGB').resize((100, 100)))
    img_data = image_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    kmeans.fit(img_data)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    hsv_colors = [rgb_to_hsv(tuple(rgb)) for rgb in dominant_colors]
    return {
        'rgb': [tuple(c) for c in dominant_colors],
        'hsv': hsv_colors
    }

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

def score_grid(images):
    palettes = [extract_color_palette(img) for img in images]
    return calculate_harmony_score(palettes)

def create_grid_image(images, grid_size=(3, 3), image_size=(200, 200)):
    rows, cols = grid_size
    grid_img = Image.new('RGB', (cols * image_size[0], rows * image_size[1]))

    for idx, img in enumerate(images):
        resized = img.resize(image_size)
        row, col = divmod(idx, cols)
        grid_img.paste(resized, (col * image_size[0], row * image_size[1]))

    return grid_img

def find_top_layouts(all_images, sample_size=9, max_attempts=100):
    if len(all_images) < sample_size:
        return []

    seen = set()
    top_layouts = []

    for _ in range(max_attempts):
        candidate = tuple(random.sample(all_images, sample_size))
        if candidate in seen:
            continue
        seen.add(candidate)

        score = score_grid(candidate)
        top_layouts.append((score, list(candidate)))
        top_layouts = sorted(top_layouts, key=lambda x: -x[0])[:3]  # keep top 3

    return top_layouts

# ---------- STREAMLIT APP ---------- #

st.set_page_config(page_title="Grid Harmony Optimizer", layout="wide")
st.title("🎨 Choose Your Most Harmonious 9-Image Grid")

uploaded_files = st.file_uploader("Upload at least 9 images", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 9:
    st.success(f"{len(uploaded_files)} images uploaded successfully.")
    images = [Image.open(file).convert("RGB") for file in uploaded_files]

    if st.button("🔍 Find Best 9-Image Grid Layouts"):
        with st.spinner("Analyzing color harmony..."):
            top_results = find_top_layouts(images, sample_size=9, max_attempts=150)

        if top_results:
            st.subheader("Top Layout Suggestions")
            for idx, (score, layout) in enumerate(top_results, 1):
                grid_img = create_grid_image(layout)
                st.image(grid_img, caption=f"Layout {idx} - Harmony Score: {score}", use_column_width=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    grid_img.save(tmp.name)
                    st.download_button(
                        label=f"Download Layout {idx}",
                        data=open(tmp.name, "rb"),
                        file_name=f"grid_layout_{idx}.jpg",
                        mime="image/jpeg",
                        key=f"download_{idx}"
                    )
        else:
            st.error("Could not generate layouts. Try uploading more diverse images.")
else:
    st.info("Please upload at least 9 images to begin.")
