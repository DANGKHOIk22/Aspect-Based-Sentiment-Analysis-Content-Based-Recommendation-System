import os
import torch
import numpy as np
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model, get_peft_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import evaluate
import warnings
import bitsandbytes as bnb
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding
from typing import Any, Dict, List
from datasets import DatasetDict
warnings.filterwarnings("ignore")



def get_base_model_config():
    return "meta-llama/Llama-3.2-1B-Instruct", "./cache"

def get_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

def get_training_args():
    return TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=400,
        max_steps=5100,
        eval_steps=400,
        eval_strategy="steps",
        overwrite_output_dir=True,
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
    )
def load_model_and_tokenizer(base_model_id, cache_dir, quant_config):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, cache_dir=cache_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
        quantization_config=quant_config,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, base_model
def get_lora_model(base_model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    return get_peft_model(base_model, peft_config)
def prepare_dataset():
    ds = load_dataset("thainq107/abte-restaurants")
    label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
    id2label = {v: k for k, v in label2id.items()}

    def align_labels(examples):
        labels = []
        for tokens, pols in zip(examples['Tokens'], examples['Polarities']):
            pols_label = next((int(p) for i, p in enumerate(pols) if int(p) != -1), 0)
            labels.append(pols_label)
        return {'labels': torch.tensor(labels)}

    ds = ds.map(align_labels, batched=True, desc="Aligning labels")
    return ds, label2id, id2label
def get_tokenize_function(tokenizer, id2label):
    USER_PROMPT_TEMPLATE = """Predict the sentiment of the following input sentence.
The response must begin with "Sentiment: ", followed by one of these keywords: "positive", "negative", or "neutral", to reflect the sentiment of the input sentence.

Sentence: {input}"""

    def tokenize_function(examples):
        results = {"input_ids": [], "labels": [], "attention_mask": []}
        for tokens, label in zip(examples['Tokens'], examples['labels']):
            prompt = USER_PROMPT_TEMPLATE.format(input=tokens)
            response = f"Sentiment: {id2label[label]}"

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            full_convo = messages + [{"role": "assistant", "content": response}]

            input_tokens = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)[0]
            output_tokens = tokenizer.apply_chat_template(full_convo, return_tensors="pt")[0]

            input_ids = output_tokens
            label_ids = torch.cat([
                torch.full_like(input_tokens, -100),
                output_tokens[len(input_tokens):]
            ])

            results["input_ids"].append(input_ids)
            results["labels"].append(label_ids)
            results["attention_mask"].append(torch.ones_like(input_ids))
        return results

    return tokenize_function
class RightPaddingDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids, labels, attention_mask = [], [], []
        max_len = max(len(f["input_ids"]) for f in features)

        for f in features:
            ids = torch.tensor(f["input_ids"], dtype=torch.long)
            lbls = torch.tensor(f["labels"], dtype=torch.long)
            mask = torch.ones_like(ids)

            pad_len = max_len - len(ids)
            ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            lbls = torch.cat([lbls, torch.full((pad_len,), -100)])
            mask = torch.cat([mask, torch.zeros(pad_len)])

            input_ids.append(ids[:max_len])
            labels.append(lbls[:max_len])
            attention_mask.append(mask[:max_len])

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask)
        }
def get_compute_metrics_fn(tokenizer, label2id):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def postprocess_logits(logits, labels):
        return logits[0].argmax(dim=-1) if isinstance(logits, tuple) else logits.argmax(dim=-1)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions

        idx = next(i for i in range(len(predictions[0])) if labels[0][i] != -100)
        predictions = predictions[:, idx:]
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        def decode(seq):
            return tokenizer.batch_decode(seq, skip_special_tokens=True)

        pred_texts = decode([p[:np.where(p == tokenizer.eos_token_id)[0][0]] if tokenizer.eos_token_id in p else p for p in predictions])
        label_texts = decode(labels)

        def extract(sent): return label2id.get(sent.split(":")[-1].strip(), 1)
        pred_ids = [extract(p) for p in pred_texts]
        label_ids = [extract(l) for l in label_texts]

        return {
            **accuracy.compute(predictions=pred_ids, references=label_ids),
            **f1.compute(predictions=pred_ids, references=label_ids, average="macro"),
            **precision.compute(predictions=pred_ids, references=label_ids, average="macro"),
            **recall.compute(predictions=pred_ids, references=label_ids, average="macro")
        }

    return postprocess_logits, compute_metrics
def train_model():
    base_model_id, cache_dir = get_base_model_config()
    tokenizer, base_model = load_model_and_tokenizer(base_model_id, cache_dir, get_quant_config())
    peft_model = get_lora_model(base_model)

    raw_ds, label2id, id2label = prepare_dataset()
    tokenize_fn = get_tokenize_function(tokenizer, id2label)

    col_names = raw_ds['train'].column_names
    tokenized_dataset = raw_ds.map(tokenize_fn, batched=True, remove_columns=col_names, num_proc=os.cpu_count())

    
    valid_split = tokenized_dataset["test"].train_test_split(test_size=0.5, seed=42)
    tokenized_dataset = DatasetDict({
        "train": tokenized_dataset["train"],
        "validation": valid_split["train"],
        "test": valid_split["test"]
    })

    collator = RightPaddingDataCollator(tokenizer)
    postprocess_fn, compute_fn = get_compute_metrics_fn(tokenizer, label2id)

    trainable = filter(lambda p: p.requires_grad, peft_model.parameters())
    optimizer = bnb.optim.PagedAdamW(trainable, lr=3e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(5100*0.1), num_training_steps=5100)

    trainer = SFTTrainer(
        model=peft_model,
        args=get_training_args(),
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=collator,
        preprocess_logits_for_metrics=postprocess_fn,
        compute_metrics=compute_fn,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
