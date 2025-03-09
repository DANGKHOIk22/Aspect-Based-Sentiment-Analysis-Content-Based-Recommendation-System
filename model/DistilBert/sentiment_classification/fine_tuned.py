import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer
import os
from transformers import AutoModelForTokenClassification
from preprocess import tokenize
from preprocess import preprocess_pipeline

accuracy = evaluate.load("accuracy")
id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
label2id = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

def initialize_model():
    """Initialize the token classification model with specified configurations."""
    return AutoModelForTokenClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def setup_training(preprocessed_ds):
    os.environ['WANDB_DISABLED'] = 'true'
    training_args = TrainingArguments(
    output_dir="absa-restaurants-albert-base-v2",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=50,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # report_to="wandb",
    )
    model = initialize_model()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_ds["train"],
        eval_dataset=preprocessed_ds["test"],
        processing_class=tokenize(),
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