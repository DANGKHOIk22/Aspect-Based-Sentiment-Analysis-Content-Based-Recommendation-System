import streamlit as st
from api import predict
import pandas as pd
import cv2
import sys, os
from PIL import Image
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)


st.title("📊 Aspect-based Sentiment Analysis and Recommend System")
data_path = os.path.join(PROJECT_ROOT, "data", "final_data.csv")
processed_data_path = os.path.join(PROJECT_ROOT, "data", "processed_data.csv")

data = pd.read_csv(data_path)
processed_data = pd.read_csv(processed_data_path, index_col=0)

# Book title dropdown
st.markdown("<h1 style='text-align: center;'>📚 Book Explorer</h1>", unsafe_allow_html=True)
option = st.selectbox("Search", data['book_title'].dropna().unique())

# Selected book info
selected = data[data['book_title'] == option].iloc[0]
book_id = selected['book_id']
image_path = os.path.join(PROJECT_ROOT, "image", f"{book_id}.jpg")

col1, col2 = st.columns([1, 2])

with col1:
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=option, use_container_width=True)
    else:
        st.warning("Image not found.")

with col2:
    st.markdown(f"<h2 style='margin-bottom: 0;'>{option}</h2>", unsafe_allow_html=True)
    st.markdown(f"**Genre:** {selected.get('genre', 'N/A')}")
    st.markdown(f"**Pages:** {selected.get('pages', 'N/A')}")
    st.markdown(f"**Rating:** <span style='background-color:#2e7d32; color:white; padding: 4px 8px; border-radius: 5px;'>{selected.get('rating', 'N/A')}</span>", unsafe_allow_html=True)
    st.markdown(f"**Ratings Count:** {selected.get('ratings_count', 'N/A')}")
    st.markdown(f"**Reviews Count:** {selected.get('reviews_count', 'N/A')}")

comment_input = st.text_area("Enter a comment:")



if comment_input.strip():
    results = predict(book_id,comment_input)
    recommend_ids = results["recommend_ids"]
    sentiment_label = results["sentiment_label"]

    # Map sentiment to colors
    sentiment_colors = {
        "Positive": "#2e7d32",   # green
        "Neutral": "#616161",    # gray
        "Negative": "#c62828"    # red
    }
    sentiment_color = sentiment_colors.get(sentiment_label, "#757575")  # default gray

    # Show sentiment label with color
    st.markdown(f"""
        <div style='text-align:center; margin-top:20px;'>
            <span style='font-size:18px;'>Sentiment of your comment:</span><br>
            <span style='background-color:{sentiment_color}; color:white; padding:6px 12px; border-radius:6px; font-size:20px;'>
                {sentiment_label}
            </span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📘 Recommended Books")
    for rec_id in recommend_ids:
        rec = data[data['book_id'] == rec_id].iloc[0]
        rec_image_path = os.path.join(PROJECT_ROOT, "image", f"{rec_id}.jpg")

        col1, col2 = st.columns([1, 2])
        with col1:
            if os.path.exists(rec_image_path):
                rec_image = Image.open(rec_image_path)
                st.image(rec_image, caption=rec['book_title'], use_container_width=True)
            else:
                st.warning("Image not found.")

        with col2:
            st.markdown(f"<h2 style='margin-bottom: 0;'>{rec['book_title']}</h2>", unsafe_allow_html=True)
            st.markdown(f"**Genre:** {rec.get('genre', 'N/A')}")
            st.markdown(f"**Pages:** {rec.get('pages', 'N/A')}")
            st.markdown(f"**Rating:** <span style='background-color:#2e7d32; color:white; padding: 4px 8px; border-radius: 5px;'>{rec.get('rating', 'N/A')}</span>", unsafe_allow_html=True)
            st.markdown(f"**Ratings Count:** {rec.get('ratings_count', 'N/A')}")
            st.markdown(f"**Reviews Count:** {rec.get('reviews_count', 'N/A')}")
