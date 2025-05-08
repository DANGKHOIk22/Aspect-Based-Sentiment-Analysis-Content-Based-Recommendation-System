from fastapi import HTTPException, status
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os, sys
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..', '..'))
sys.path.append(PROJECT_ROOT)
# Load pipelines
token_classifier = pipeline(
    model="Khoivudang1209/abte-restaurants-distilbert-base-uncased",
    aggregation_strategy="simple"
)
sentiment_classifier = pipeline(
    model="Khoivudang1209/abte-restaurants-sentiment-distilbert"
)

sentiment_to_ids = {
    'Positive': 1,
    'Neutral': 0,
    'Negative': -1
}
def recommend_system(book_id, comment, processed_data, data, top_k=5):
    try:
        aspects = token_classifier(comment)
        aspects = " ".join([aspect['word'] for aspect in aspects if aspect['score'] > 0.45])
        sentiment_result = sentiment_classifier(f'{comment} [SEP] {aspects}')
        sentiment_label = sentiment_result[0]['label']
        sentiment_val = sentiment_to_ids.get(sentiment_label, 0)

        input_vector = np.concatenate([np.array(processed_data.loc[book_id])[:-1], [sentiment_val / 2]])
        similarities = cosine_similarity([input_vector], processed_data)[0]
        top_indices = similarities.argsort()[::-1]
        top_indices = [i for i in top_indices if i != book_id][:top_k]
        return data[data['book_id'].isin(top_indices)]['book_id'].tolist(), sentiment_label 
    except Exception as e:
        return [], None

async def predict_label(input_data):
    try:
        data_path = os.path.join(PROJECT_ROOT, "data", "final_data.csv")
        processed_data_path = os.path.join(PROJECT_ROOT, "data", "processed_data.csv")
        print(1111)
        data = pd.read_csv(data_path)
        processed_data = pd.read_csv(processed_data_path, index_col=0)
        recommend_ids, sentiment_label = recommend_system(input_data.book_id, input_data.comment, processed_data, data)
        return {
            'recommend_ids': recommend_ids,
            'sentiment_label':  sentiment_label
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing sentiment classification: {str(e)}"
        )