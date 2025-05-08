from transformers import pipeline
import pandas as pd
import torch
import os,sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
# Load pre-trained models
token_classifier = pipeline(
    model="Khoivudang1209/abte-restaurants-distilbert-base-uncased",
    aggregation_strategy="simple"
)
sentiment_classifier = pipeline(
    model="Khoivudang1209/abte-restaurants-sentiment-distilbert"
)

# Mapping for sentiment label to numerical representation
sentiment_to_ids = {
    'Positive': 1,
    'Neutral': 0,
    'Negative': -1
}


def extract_aspects(data, token_classifier, batch_size=32, score_threshold=0.45):
    all_comments = data['comment_text'].tolist()
    aspects = []

    for i in range(0, len(all_comments), batch_size):
        batch = all_comments[i:i + batch_size]
        results = token_classifier(batch)
        for result in results:
            aspect_words = ' '.join([word['word'] for word in result if word['score'] > score_threshold])
            aspects.append(aspect_words)

    data.loc[:, 'aspect'] = aspects
    return data


def classify_sentiment(data, sentiment_classifier, batch_size=32):
    all_comments = data['comment_text'].tolist()
    all_aspects = data['aspect'].tolist()
    sentiment = []

    for i in range(0, len(all_comments), batch_size):
        batch_comment = all_comments[i:i + batch_size]
        batch_aspect = all_aspects[i:i + batch_size]
        inputs = [f"{c} [SEP] {a}" for c, a in zip(batch_comment, batch_aspect)]
        results = sentiment_classifier(inputs)
        batch_sentiments = [r['label'] for r in results]
        sentiment.extend(batch_sentiments)

    data.loc[:, 'sentiment'] = sentiment
    return data


def compute_sentiment_scores(data):
    data_onehot = pd.get_dummies(data['sentiment'])
    data_combined = pd.concat([data['book_title'], data_onehot], axis=1)

    sentiment_counts = data_combined.groupby('book_title').sum()
    sentiment_counts['sentiment_score'] = (
        sentiment_counts.get('Positive', 0) * 1 +
        sentiment_counts.get('Neutral', 0) * 0 +
        sentiment_counts.get('Negative', 0) * -1
    ) / (
        sentiment_counts.get('Positive', 0) +
        sentiment_counts.get('Neutral', 0) +
        sentiment_counts.get('Negative', 0)
    )

    sentiment_counts = sentiment_counts.reset_index()
    return sentiment_counts


def merge_and_finalize(data, sentiment_counts):
    data_merged = data.merge(sentiment_counts, on='book_title', how='left')
    data_merged = data_merged.drop(['comment_text', 'aspect', 'sentiment', 'Positive', 'Negative', 'Neutral'], axis=1)
    data_merged = data_merged.drop_duplicates()

    # One-hot encode genres
    one_hot = pd.crosstab(data_merged['book_title'], data_merged['genre']).reset_index()
    data_merged = data_merged.drop(['genre'], axis=1)
    data_merged = data_merged.drop_duplicates()
    data_merged = data_merged.merge(one_hot, on='book_title')
    data_merged = data_merged.drop_duplicates(subset='book_title', keep='first')
    data_merged = data_merged.drop(['book_title'], axis=1)

    # Reorder columns
    cols = [col for col in data_merged.columns if col != 'sentiment_score'] + ['sentiment_score']
    data_merged = data_merged[cols]

    return data_merged


def preprocess_pipeline(data, token_classifier, sentiment_classifier, batch_size=32):
    processed_data = data.copy()
    processed_data = extract_aspects(processed_data, token_classifier, batch_size)
    processed_data = classify_sentiment(processed_data, sentiment_classifier, batch_size)
    sentiment_summary = compute_sentiment_scores(processed_data)
    processed_data = merge_and_finalize(processed_data, sentiment_summary)

    processed_data.sort_values(by='book_id', inplace=True)
    processed_data.set_index('book_id', inplace=True)
    return processed_data


def recommend_system(book_id, comment, processed_data, data, top_k=5):
    aspects = token_classifier(comment)
    aspects = " ".join([aspect['word'] for aspect in aspects if aspect['score'] > 0.45])
    
    sentiment_result = sentiment_classifier(f'{comment} [SEP] {aspects}')
    sentiment_label = sentiment_result[0]['label']

    sentiment_val = sentiment_to_ids[sentiment_label]
    input_vector = np.concatenate([np.array(processed_data.loc[book_id])[:-1], [sentiment_val / 2]])

    similarities = cosine_similarity([input_vector], processed_data)[0]
    top_indices = similarities.argsort()[::-1]
    top_indices = [i for i in top_indices if i != book_id][:top_k]

    return data.loc[top_indices]


def main():
    # Load and clean data
    path = os.path.join(PROJECT_ROOT, "data","data.csv")

    data = pd.read_csv(path)
    data.sort_values(by='book_id', inplace=True)
    data = data.dropna()
    # Preprocess for model input
    processed_data = preprocess_pipeline(data, token_classifier, sentiment_classifier)

    # Clean and align metadata
    data = data.drop(['comment_text'], axis=1)
    data = data.drop_duplicates()
    data_genre = data.groupby('book_title')['genre'].agg(list).reset_index()
    data = data.drop('genre', axis=1)
    data = data.drop_duplicates(subset='book_title', keep='first')
    data = data.merge(data_genre, on='book_title')
    data.set_index('book_id', inplace=True)

    processed_data.to_csv('processed_data.csv', index=True)
    data.to_csv('final_data.csv', index=True)


