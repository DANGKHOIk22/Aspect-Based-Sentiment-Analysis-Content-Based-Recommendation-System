import evaluate
import numpy as np
from model import ABTELSTMClassifier, ABTEConv1DClassifier, ABTETransformerClassifier
import os
from transformers import Trainer, TrainingArguments
from model.Conv1D_LSTM_Transformer.aspect_prediction.preprocess import preprocess_pipeline,tokenize
from transformers import PretrainedConfig
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def setup_training(preprocessed_ds):
    os.environ['WANDB_DISABLED'] = 'true'
    model_name = ['lstm','conv1d','transformer']
    training_args = TrainingArguments(
        output_dir=f"abte-restaurants-{model_name}-base-uncased",
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
        metric_for_best_model="accuracy",
    )
    config = PretrainedConfig()
    tokenizer = tokenize()
    model = ABTELSTMClassifier(config, len(tokenizer.get_vocab()), num_classes=3)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_ds["train"],
        eval_dataset=preprocessed_ds["test"],
        tokenizer=tokenizer,
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