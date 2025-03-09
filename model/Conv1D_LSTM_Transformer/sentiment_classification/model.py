import torch
import torch.nn as nn
from transformers import PreTrainedModel

class ABSALSTMClassifier(PreTrainedModel):
    def __init__(self, config, vocab_size, num_classes,
                 embedding_dim=256, hidden_dim=256, pad_idx=0, dropout=0.3):
        super().__init__(config)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,  batch_first=True,bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim*2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.norm = nn.LayerNorm(hidden_dim*2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids)  # Shape: (batch_size, seq_len, embedding_dim)
        lstm_output, (hidden, _) = self.lstm(embedded)  # hidden: (num_layers*2, batch_size, hidden_dim)
        
        # For bidirectional LSTM, concatenate the last forward and first backward hidden states
        # hidden shape: (num_layers * directions, batch_size, hidden_dim)
        # Take the last layer's hidden states for both directions
        hidden_forward = hidden[-2]  # Last forward layer
        hidden_backward = hidden[-1]  # Last backward layer
        # Concatenate along the hidden dimension
        hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)  # Shape: (batch_size, hidden_dim * 2)
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)  # Apply dropout
        hidden = self.fc1(hidden)  # Shape: (batch_size, num_classes)
        logits = self.fc2(hidden)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)  # labels should have shape (batch_size,)
        
        return {"loss": loss, "logits": logits}
    
class ABSAConv1DClassifier(PreTrainedModel):
    def __init__(self, config, vocab_size, num_classes,
                 embedding_dim=256, num_filters=256, kernel_sizes=[3,4,5],
                 pad_idx=0, dropout=0.3):
        super().__init__(config)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Fixed: Use ModuleList properly for multiple conv layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=1
            ) for k in kernel_sizes
        ])
        
        # Removed: redundant single Conv1d layer that was overwriting the ModuleList
        # Added: Calculate the correct input size for fc layer
        # Since we have len(kernel_sizes) filters and we max pool, we multiply num_filters by len(kernel_sizes)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)  # Added: dropout layer
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids)  # (B, S, E)
        embedded = embedded.permute(0, 2, 1)  # (B, E, S) for Conv1d

        # Fixed: Apply convolutions and pooling correctly
        conv_outputs = [torch.relu(conv(embedded)) for conv in self.convs]  # List of (B, num_filters, S')
        pooled = [torch.max(conv, dim=2).values for conv in conv_outputs]  # List of (B, num_filters)
        x = torch.cat(pooled, dim=1)  # (B, num_filters * len(kernel_sizes))
        
        x = self.dropout(x)  # Apply dropout
        logits = self.fc(x)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)  # labels should be (B,)

        return {"loss": loss, "logits": logits}


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=512):
        super().__init__()
        # Create a matrix of shape (max_len, num_hiddens)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_hiddens, 2) * (-torch.log(torch.tensor(10000.0)) / num_hiddens))
        
        pe = torch.zeros(max_len, num_hiddens)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, num_hiddens)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_hiddens)
        x = x + self.pe[:x.size(0)]
        return x

class ABTETransformerClassifier(PreTrainedModel):
    def __init__(self, config, vocab_size, num_classes,
                 max_len=512, embedding_dim=256, num_heads=8,
                 num_layers=6, hidden_dim=1024, pad_idx=0,dropout =0.3):
        super().__init__(config)
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = PositionalEncoding(
            num_hiddens=embedding_dim, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            batch_first=True  # Added for better clarity and compatibility
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, input_ids, labels=None, attention_mask=None):
        # Embedding layer
        embedding = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Add positional encoding
        embedding = self.position_embedding(embedding)
        
        # Transformer expects (seq_len, batch_size, embedding_dim) if batch_first=False
        # Since we use batch_first=True, we keep (batch_size, seq_len, embedding_dim)

        encoded = self.transformer_encoder(
            embedding, 
        )
        encoded = self.norm(encoded)
        # Get CLS representation (first token)
        
        # encoded[:, 0, :]
        cls_representation = encoded[:, 0, :]  # (batch_size, embedding_dim)
        cls_representation  = self.dropout(cls_representation)
        # Final classification layer
        logits = self.fc(cls_representation)  # (batch_size, num_classes)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}