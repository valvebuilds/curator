import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import cv2
import pandas as pd

# Setup model
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    return model, processor

model, processor = load_model()

# Theme prompts
all_themes = [
    "dreamy nature scene", "cinematic street shadows", "geometric colorful portraits",
    "vintage aesthetics", "urban decay", "surrealism and fantasy", "intimate close-up",
    "melancholic blue hour", "high-contrast monochrome drama", "warm golden nostalgia",
    "neon-drenched nightscape", "ritual and celebration", "fog and silence",
    "pastel minimalism", "chaotic street energy", "introspective solitude",
    "ethereal underwater world", "decaying grandeur", "seasonal transitions",
    "backstage moments", "color splash rebellion", "ritualistic light play"
]

st.subheader("🎨 Select Preferred Themes")
selected_themes = st.multiselect(
    "Tap your favorite themes to personalize the classification.",
    options=all_themes,
    default=["dreamy nature scene", "vintage aesthetics", "surrealism and fantasy"]
)

if not selected_themes:
    st.warning("Please select at least one theme to proceed.")
    st.stop()


# Precompute theme embeddings
theme_inputs = processor.tokenizer(
    selected_themes,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    theme_embeddings = model.get_text_features(**theme_inputs)
    theme_embeddings = theme_embeddings / theme_embeddings.norm(dim=1, keepdim=True)

def get_cluster_theme_label(image_features_cluster):
    cluster_embedding = image_features_cluster.mean(dim=0)
    cluster_embedding = cluster_embedding / cluster_embedding.norm()
    similarities = torch.matmul(theme_embeddings, cluster_embedding)
    top_idx = similarities.argmax().item()
    return selected_themes[top_idx]

# UI
st.title("Viewfinder: AI Image Curation")
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# Simulate theme grid
st.markdown("**Click below to select from the following themes:**")
cols = st.columns(3)
for i, theme in enumerate(all_themes):
    with cols[i % 3]:
        if theme in selected_themes:
            st.markdown(f"✅ **{theme}**", unsafe_allow_html=True)
        else:
            st.markdown(f"⬜ {theme}", unsafe_allow_html=True)
            
if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} images. Processing...")

    images = []
    filenames = []
    embeddings = []
    scores = []

    for file in uploaded_files:
        try:
            image = Image.open(file).convert("RGB")
            image = image.resize((224, 224))  # Resize for CLIP compatibility
            images.append(image)
            filenames.append(file.name)

            inputs = processor(images=[image], return_tensors="pt")
            with torch.no_grad():
                embed = model.get_image_features(**inputs)
            embeddings.append(embed[0].numpy())

            np_img = np.array(image)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(lap)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if len(embeddings) < 3:
        st.warning("At least 3 images are required for clustering.")
    else:
        # Clustering
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Group embeddings by cluster
        embeddings_tensor = torch.tensor(embeddings)
        cluster_theme_labels = []
        for cluster in range(k):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
            cluster_embeddings = embeddings_tensor[cluster_indices]
            theme_label = get_cluster_theme_label(cluster_embeddings)
            cluster_theme_labels.append(theme_label)

        # Display images per cluster with theme
        st.subheader("Clustered Images by Theme")
        for cluster in range(k):
            st.markdown(f"### Cluster {cluster + 1} — **{cluster_theme_labels[cluster]}**")
            cols = st.columns(3)
            cluster_imgs = [
                (img, name, score)
                for img, name, label, score in zip(images, filenames, labels, scores)
                if label == cluster
            ]
            for i, (img, name, score) in enumerate(cluster_imgs):
                with cols[i % 3]:
                    st.image(img, caption=f"{name} | Score: {round(score, 2)}", use_container_width=True)

        # Export results
        if st.button("Export Results CSV"):
            df = pd.DataFrame({
                "filename": filenames,
                "cluster": labels,
                "theme": [cluster_theme_labels[label] for label in labels],
                "quality_score": scores
            })
            df.to_csv("results.csv", index=False)
            st.success("results.csv saved in app folder.") 