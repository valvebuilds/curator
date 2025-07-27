import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from itertools import combinations
from PIL import Image

# SETTINGS
NUM_IMAGES_TO_SELECT = 9
PALETTE_SIZE = 5
IMAGE_DIR = "your_image_folder_path"

# --- 1. Helper Functions ---
def resize_image(img, size=(200, 200)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def extract_palette(img, k=5):
    img = resize_image(img)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(img)
    return kmeans.cluster_centers_

def rgb_to_lab(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    return convert_color(sRGBColor(r, g, b), LabColor)

def average_palette_distance(palette1, palette2):
    distances = []
    for color1 in palette1:
        lab1 = rgb_to_lab(color1)
        for color2 in palette2:
            lab2 = rgb_to_lab(color2)
            distances.append(delta_e_cie2000(lab1, lab2))
    return np.mean(distances)

# --- 2. Load Images and Extract Palettes ---
def load_images_and_palettes(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    data = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        palette = extract_palette(img, k=PALETTE_SIZE)
        data.append({
            "path": img_path,
            "palette": palette
        })
    return data

# --- 3. Score Combinations of Images ---
def score_combination(images_subset):
    total_distance = 0
    for img1, img2 in combinations(images_subset, 2):
        dist = average_palette_distance(img1["palette"], img2["palette"])
        total_distance += dist
    return total_distance

# --- 4. Select Optimal Grid ---
def select_best_grid(images_data, select_count=9):
    from random import sample
    best_score = -np.inf
    best_combination = None
    trials = 3000  # adjustable: balance between speed and quality

    for _ in range(trials):
        candidates = sample(images_data, select_count)
        score = score_combination(candidates)
        if score > best_score:
            best_score = score
            best_combination = candidates
    return best_combination

# --- 5. Generate and Save Grid ---
def save_grid(selected_images, output_path="grid_output.jpg", cell_size=(300, 300)):
    rows = cols = int(np.sqrt(len(selected_images)))
    grid_img = Image.new("RGB", (cell_size[0]*cols, cell_size[1]*rows))

    for idx, img_data in enumerate(selected_images):
        img = Image.open(img_data["path"]).convert("RGB")
        img = img.resize(cell_size)
        x = (idx % cols) * cell_size[0]
        y = (idx // cols) * cell_size[1]
        grid_img.paste(img, (x, y))

    grid_img.save(output_path)
    print(f"Saved grid to: {output_path}")

# --- 6. Full Pipeline ---
if __name__ == "__main__":
    print("Loading and analyzing images...")
    all_images = load_images_and_palettes(IMAGE_DIR)

    if len(all_images) < NUM_IMAGES_TO_SELECT:
        raise ValueError("Not enough images to select from.")

    print(f"Selecting best {NUM_IMAGES_TO_SELECT} images based on color diversity and harmony...")
    best_images = select_best_grid(all_images, select_count=NUM_IMAGES_TO_SELECT)

    save_grid(best_images)
