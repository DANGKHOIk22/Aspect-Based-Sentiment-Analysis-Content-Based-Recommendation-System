# Import statements
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch
import evaluate
import numpy as np
import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '....'))
sys.path.append(PROJECT_ROOT)

def initialize_token_classifier():
    """Initialize the token classifier pipeline"""
    return pipeline(
        model="Khoivudang1209/abte-restaurants-distilbert-base-uncased",
        aggregation_strategy="simple"
    )

def extract_aspects(data, token_classifier):
    """Extract aspects from comments using token classifier"""
    aspects = []
    for i in range(len(data)):
        result = token_classifier(data['Comment'][i])
        aspect_words = ' '.join([word['word'] for word in result])
        aspects.append([aspect_words])
    return aspects

def clean_data(data):
    """Remove rows with empty aspects"""
    empty_indices = [i for i in range(len(data)) if data['aspect'][i][0] == '']
    return data.drop(empty_indices)

def convert_to_dataset(data):
    """Convert pandas DataFrame to DatasetDict with train/test split"""
    data['Comment'] = data['Comment'].apply(lambda x: x.split(' '))
    data['aspect'] = data['aspect'].apply(lambda x: x[0].split(' '))
    data['labels'] = data['Sentiment'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0})
    
    train_df, test_df = train_test_split(data, test_size=0.237, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return dataset_dict.remove_columns(['__index_level_0__'])

def get_max_length(dataset):
    """Calculate maximum length for tokenization"""
    max_len_tokens = max([len(tokens) for tokens in dataset['train']['Comment']])
    max_len_tags = max([len(tokens) for tokens in dataset['train']['aspect']])
    return max_len_tokens + max_len_tags

def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize and align labels for model input"""
    sentence_token = []
    sentence_aspect = []
    
    for tokens, aspects in zip(examples['Comment'], examples['aspect']):
        bert_token = []
        bert_aspect = []
        
        for token in tokens:
            bert_token += tokenizer(token)
        for aspect in aspects:
            bert_aspect += tokenizer(aspect)
            
        sentence_token.append(' '.join(bert_token))
        sentence_aspect.append(' '.join(bert_aspect))
    
    tokenized_inputs = tokenizer(
        sentence_token,
        sentence_aspect,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

def compute_metrics(eval_pred):
    """Compute accuracy metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    # Initialize components
    token_classifier = initialize_token_classifier()
    tokenizer = AutoTokenizer.from_pretrained("Khoivudang1209/abte-restaurants-sentiment-distilbert")
    path ='D:\CODE\AirFlow\data\data.csv'
    # Load and process data
    data = pd.read_csv('/kaggle/input/data-pretrained-sentiment/data.csv')
    data['aspect'] = extract_aspects(data, token_classifier)
    data = clean_data(data)
    
    # Convert to dataset
    ds = convert_to_dataset(data)
    MAX_LEN = get_max_length(ds)
    
    # Preprocess dataset
    preprocessed_ds = ds.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True
    )
    
    # Setup evaluation
    os.environ['WANDB_DISABLED'] = 'true'
    global accuracy
    accuracy = evaluate.load("accuracy")
    
    # Model configuration
    id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    label2id = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "Khoivudang1209/abte-restaurants-sentiment-distilbert",
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir="absa-restaurants-albert-base-v2",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # Initialize and run trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_ds["train"],
        eval_dataset=preprocessed_ds["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()