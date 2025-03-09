# Project-Aspect-Sentiment-Analysis

# Overview:

This project uses DistilBERT, Conv1D, LSTM, and Transformer models for aspect prediction and sentiment classification. It then utilizes DistilBERT to predict the aspect of the scraped data, fine-tunes it, and performs sentiment classification.
# Workflow:

# Dataset:
  - [Hugging Face Dataset](https://huggingface.co/datasets/thainq107/abte-restaurants)
  - Scarping data from [Goodreads](https://www.goodreads.com/)

# Main Folders:
  - **`dags/`**: 
       - `etl_pipeline.py` : Execute ETL 
  - **`image/`**: workflow image
  - **`model/`**: 
       - `Conv1D_LSTM_Transformer/`
           - `aspect_prediction/` :
               -  `model.py`: 
               - `preprocess.py` :
               - `train.py`:
           - `sentiment_classification/`:
               -  `model.py`: 
               - `preprocess.py` :
               - `train.py`:
        - `DistilBert/`
           - `aspect_prediction/` :
               -  `model.py`: 
               - `preprocess.py` :
               - `fine_tuned.py`:
           - `sentiment_classification/`:
               -  `model.py`: 
               - `preprocess.py` :
               - `fine_tuned.py`:
               - `fine_tuned_with_scraped_data.py`:
    - **`plugins/`**:
        - `postgresql_operator.py`:
        - `preprocessing.py`:
        - `scaping_data.py`:

# Results:
  - `aspect_prediction`:
     - `Conv1D` :[Hugging_face](https://huggingface.co/Khoivudang1209/abte-restaurants-conv1d)
     - `LSTM` :[Hugging_face](https://huggingface.co/Khoivudang1209/abte-restaurants-lstm)
     - `Transformer` :[Hugging_face](https://huggingface.co/Khoivudang1209/abte-restaurants-transformer)
     - `DistilBert` :[Hugging_face](https://huggingface.co/Khoivudang1209/abte-restaurants-distilbert-base-uncased)
  - `sentiment_classification`:
     - `Conv1D` :[Hugging_face](https://huggingface.co/Khoivudang1209/abte-restaurants-sentiment-conv1d)
     - `LSTM` :[Hugging_face](https://huggingface.co/Khoivudang1209/abte-restaurants-sentiment-lstm)
     - `Transformer` :[Hugging_face](https://huggingface.co/Khoivudang1209/abte-restaurants-sentiment-transformer)
     - `DistilBert` :[Hugging_face](https://huggingface.co/Khoivudang1209/abte-restaurants-sentiment-distilbert)
  - `sentiment_classification fine-tuned with scraped_data`:
     - `DistilBert`:[Hugging_face](https://huggingface.co/Khoivudang1209/absa-restaurants-albert-base-v2)

     
