from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
import torch

def tokenize():
    # Initialize the tokenizer using WordLevel model (word-based tokenization)
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))

    # Use Whitespace pre-tokenizer to split words based on spaces
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Train the tokenizer on the dataset
    trainer = trainers.WordLevelTrainer(vocab_size=5000,special_tokens=["<pad>", "<unk>"])
    tokenizer.train_from_iterator(corpus, trainer)

    # Save the trained tokenizer as a JSON file
    tokenizer.save("word_tokenizer.json")
    return tokenizer
import torch

def pad_and_truncate(inputs, pad_id):
    if len(inputs) < MAX_LEN:
        padded_inputs = inputs + [pad_id] * (MAX_LEN - len(inputs))
    else:
        padded_inputs = inputs[:MAX_LEN]
    return padded_inputs


def tokenize_and_align_labels(examples):
    tokenized_inputs = []
    labels = []
    for tokens, pols in zip(examples['Tokens'], examples['Polarities']):
        token_pols = []
        pols_label = 0
        for i in range(len(tokens)):
            if int(pols[i]) != -1:
                token_pols.append(tokens[i])
                pols_label = int(pols[i])
        input_ids = tokens + token_pols
        input_ids = [tokenizer.token_to_id(token.lower()) if tokenizer.token_to_id(token.lower()) else 0 for token in tokens ]
        input_ids = pad_and_truncate(input_ids, tokenizer.token_to_id("<pad>"))

        tokenized_inputs.append(input_ids)
        labels.append(pols_label)

    return {
            'input_ids': torch.tensor(tokenized_inputs),
            'labels': torch.tensor(labels)
        }
def preprocess_pipeline():
    # Load dataset
    ds = load_dataset("thainq107/abte-restaurants")
    
    # Calculate maximum sequence length
    global MAX_LEN
    MAX_LEN = max(len(tokens) for tokens in ds['train']['Tokens'])

    global corpus
    corpus = [" ".join(i) for i in ds['train']['Tokens']]

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