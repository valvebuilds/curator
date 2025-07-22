from PIL import Image
import os

image_folder = 'sample_images/'  # Place your test images here
images = []
filenames = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(os.path.join(image_folder, filename)).convert("RGB")
        images.append(img)
        filenames.append(filename)

from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        return embeddings[0].numpy()

embeddings = [get_clip_embedding(img) for img in images]

from sklearn.cluster import KMeans

k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Map image filename → cluster label
clustered = list(zip(filenames, labels))
for fname, label in clustered:
    print(f"{fname}: Cluster {label}")

import cv2
import numpy as np

def sharpness_score(img_pil):
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

quality_scores = [sharpness_score(img) for img in images]

# Sort by score
sorted_imgs = sorted(zip(filenames, quality_scores), key=lambda x: x[1], reverse=True)
print("Top 5 by sharpness:")
for fname, score in sorted_imgs[:5]:
    print(f"{fname}: {score:.2f}")
theme_prompts = ["a portrait photo", "a landscape photo", "a flatlay", "a fashion shot"]

text_inputs = processor(text=theme_prompts, return_tensors="pt", padding=True)
text_features = model.get_text_features(**text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

for i, image in enumerate(images):
    image_input = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze()
    best_idx = similarity.argmax()
    print(f"{filenames[i]}: {theme_prompts[best_idx]} ({similarity[best_idx]:.2f})")

    import matplotlib.pyplot as plt

def show_cluster(cluster_id):
    idxs = [i for i, label in enumerate(labels) if label == cluster_id]
    fig, axs = plt.subplots(1, len(idxs), figsize=(15, 5))
    for ax, i in zip(axs, idxs):
        ax.imshow(images[i])
        ax.set_title(filenames[i])
        ax.axis("off")
    plt.show()

# Show images in cluster 0
show_cluster(0)
show_cluster(1)

show_cluster(2)