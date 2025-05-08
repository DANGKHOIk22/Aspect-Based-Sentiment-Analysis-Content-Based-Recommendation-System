from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
import numpy as np
from seqeval.metrics import accuracy_score, f1_score
import os
from transformers import Trainer, TrainingArguments
from preprocess import preprocess_pipeline
from transformers import AutoTokenizer
from preprocess import tokenize


data_collator = DataCollatorForTokenClassification(tokenizer=tokenize())

id2label = {
    0: "O",
    1: "B-Term",
    2: "I-Term"
}
label2id = {
    "O": 0,
    "B-Term": 1,
    "I-Term": 2
}

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = f1_score(true_predictions, true_labels)
    return {"F1-score": results}

def initialize_model():
    """Initialize the token classification model with specified configurations."""
    return AutoModelForTokenClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

def setup_training(preprocessed_ds):
    os.environ['WANDB_DISABLED'] = 'true'
    
    training_args = TrainingArguments(
        output_dir="abte-restaurants-distilbert-base-uncased",
        logging_dir="logs",
        learning_rate=2e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=25,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="F1-score",
    )
    model = initialize_model()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_ds["train"],
        eval_dataset=preprocessed_ds["test"],
        tokenizer=tokenize(),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    return trainer
def main():
    """Main execution function combining preprocessing and training."""
    # Call preprocessing pipeline
    preprocessed_ds = preprocess_pipeline()
    print("Dataset preprocessing completed")
    
    # Setup and run training
    trainer = setup_training(preprocessed_ds)
    trainer.train()
    print("Training completed")

