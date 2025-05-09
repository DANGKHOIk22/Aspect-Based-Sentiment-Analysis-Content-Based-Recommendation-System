# 📘 Aspect-Based Sentiment Analysis & Content-Based Recommendation System

This project applies advanced NLP techniques to perform **aspect-based sentiment analysis (ABSA)** on user comments and uses the results to build a **content-based book recommendation system**. The goal is to analyze sentiment at a fine-grained level and suggest relevant books based on user preferences.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Demo](#demo)
- [Installation](#installation)

---

## 🔍 Overview

- Fine-grained sentiment analysis on comments (aspect-level classification)
- Models include Conv1D, LSTM, Transformer, DistilBERT, and LLaMA
- Book recommendation system based on comment sentiment + metadata
- Data sourced from Hugging Face and web-scraped Goodreads

<div align="center">
  <img src="Screenshot 2025-05-09 132341.png" alt="Workflow Diagram" width="600">
</div>

---

## 🏗️ Architecture

This project follows a full-stack machine learning pipeline:

- **Airflow** orchestrates the ETL process (Dockerized):
  - Extracts and processes Goodreads data
  - Cleans and formats text and metadata
  - Loads the structured output into **PostgreSQL** (Dockerized)
- **PostgreSQL** stores book and comment data
- **Model training** reads from the database and saves results locally
- **FastAPI** serves trained models via RESTful endpoints (run locally)
- **Streamlit** offers an interactive web UI for sentiment prediction and recommendations (run locally)
- **Docker** is used to containerize and manage Airflow and PostgreSQL only

---

## 📁 Dataset

### 1. [ABSA Dataset (Hugging Face)](https://huggingface.co/datasets/thainq107/abte-restaurants)
- `tokens`: Tokenized input sentences  
- `tags`: Aspect tags in IOB format  
- `polarities`: Sentiment labels (positive/neutral/negative)  

### 2. Goodreads Web-Scraped Data
- `book_id`, `book_title`, `genre`, `pages`  
- `rating`, `ratings_count`, `reviews_count`  
- `comment_text`: Raw user comment text  

---

## 🧠 Methodology

### 🔸 Aspect Sentiment Models:
- **Conv1D / LSTM / Transformer**:  
  Used for aspect tagging and comment-level sentiment classification

- **DistilBERT**:  
  Combines aspect tagging with `<SEP>` and comment-level classification for better performance

- **LLaMA-3.2-1B-Instruct**:  
  Fine-tuned with instruction-style prompts for direct sentiment inference

### 🔹 Content-Based Recommendation
Books are recommended using:
- Genre similarity  
- Page count proximity  
- Rating and review count  
- Extracted sentiment from user comments

### 🔄 ETL Pipeline (Airflow)
- Scheduled DAGs:
  - Scrape Goodreads data
  - Clean & transform comment text
  - Save to PostgreSQL
- DAGs are containerized via Docker and monitored via Airflow UI

---

## 📈 Results

| **Model**               | **Accuracy** |
|-------------------------|--------------|
| Conv1D                  | 66.49%       |
| LSTM                    | 62.56%       |
| Transformer             | 63.36%       |
| DistilBERT              | **81.50%**   |
| LLaMA-3.2-1B-Instruct   | 25.36%       |

> ✅ DistilBERT achieved the best performance for sentiment classification.

---

## 🎥 Demo

[![Watch the demo]](https://www.youtube.com/watch?v=rap2EdqjEFI)



---

## 🛠 Installation

### 🔧 Local Setup

```bash
git clone https://github.com/DANGKHOIk22/Aspect-Based-Sentiment-Analysis-Content-Based-Recommendation-System.git
pip install -r requirements.txt
