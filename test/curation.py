import streamlit as st
from PIL import Image
import numpy as np
import os
import shutil
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import cv2

# Setup model
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    return model, processor

model, processor = load_model()

# Image upload
st.title("Viewfinder: AI Image Curation")
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} images. Processing...")

    images = []
    filenames = []
    embeddings = []
    scores = []

    for file in uploaded_files:
        try:
            image = Image.open(file).convert("RGB")
            images.append(image)
            filenames.append(file.name)

            # CLIP embedding
            inputs = processor(images=[image], return_tensors="pt")  # lista de 1 imagen
            with torch.no_grad():
                embed = model.get_image_features(**inputs)
            embeddings.append(embed[0].numpy())

            # Sharpness score
            np_img = np.array(image)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(lap)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    # Clustering
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Display in 3-column grid by cluster
    st.subheader("Clustered Images")
    for cluster in range(k):
        st.write(f"### Cluster {cluster + 1}")
        cols = st.columns(3)
        cluster_imgs = [(img, name, score) for img, name, label, score in zip(images, filenames, labels, scores) if label == cluster]
        for i, (img, name, score) in enumerate(cluster_imgs):
            with cols[i % 3]:
                st.image(img, caption=f"{name} | Score: {round(score, 2)}", use_column_width=True)

    # Optional: Export results
    if st.button("Export Results CSV"):
        import pandas as pd
        df = pd.DataFrame({
            "filename": filenames,
            "cluster": labels,
            "quality_score": scores
        })
        df.to_csv("results.csv", index=False)
        st.success("results.csv saved in app folder.")
