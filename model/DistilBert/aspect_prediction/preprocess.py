from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize():
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    return tokenizer

def pad_and_truncate(inputs, pad_id):
    if len(inputs) > MAX_LEN:
        return inputs[:MAX_LEN]
    return inputs + [pad_id] * (MAX_LEN - len(inputs))

def tokenize_and_align_labels(examples):
    tokenized_inputs = []
    labels = []
    
    # Process each example in the batch
    for tokens, tags in zip(examples['Tokens'], examples['Tags']):
        bert_tokens = []
        bert_tags = []
        
        # Tokenize each word and align tags
        for token, tag in zip(tokens, tags):
            subword_tokens = tokenizer.tokenize(token)
            bert_tokens.extend(subword_tokens)
            bert_tags.extend([int(tag)] * len(subword_tokens))
        
        # Convert tokens to IDs
        bert_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        
        # Verify alignment
        assert len(bert_ids) == len(bert_tags), "Token IDs and tags length mismatch"
        
        # Pad/truncate sequences
        bert_ids = pad_and_truncate(bert_ids, pad_id=0)
        bert_tags = pad_and_truncate(bert_tags, pad_id=-100)
        
        tokenized_inputs.append(bert_ids)
        labels.append(bert_tags)
    
    return {
        "input_ids": tokenized_inputs,
        "labels": labels
    }

def preprocess_pipeline():
    # Load dataset
    ds = load_dataset("thainq107/abte-restaurants")
    
    # Calculate maximum sequence length
    global MAX_LEN
    MAX_LEN = max(len(tokens) for tokens in ds['train']['Tokens'])
    
    global tokenizer
    tokenizer = tokenize()
    
    # Apply tokenization and alignment to entire dataset
    preprocessed_ds = ds.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing and aligning labels"
    )
    
    return preprocessed_ds

# Example usage
if __name__ == "__main__":
    dataset = preprocess_pipeline()
    print("Dataset preprocessing completed")