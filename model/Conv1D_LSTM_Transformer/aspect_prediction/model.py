import torch
import torch.nn as nn
from transformers import PreTrainedModel

class ABTELSTMClassifier(PreTrainedModel):
    def __init__(self, config, vocab_size, num_classes,
                 embedding_dim=256, hidden_dim=256, pad_idx=0, dropout=0.3):
        super().__init__(config)
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,  batch_first=True,bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim*2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.norm = nn.LayerNorm(hidden_dim*2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        outputs = self.norm(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        logits = self.fc2(outputs)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1), labels)

        return {"loss": loss, "logits": logits}



class ABTEConv1DClassifier(PreTrainedModel):
    def __init__(self, config, vocab_size, num_classes,
                 embedding_dim=256, num_filters=256, kernel_size=3, pad_idx=0,dropout=0.3):
        super().__init__(config)
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv = nn.Conv1d(
            in_channels=embedding_dim, out_channels=num_filters,
            kernel_size=kernel_size, padding=1)
        self.norm = nn.LayerNorm(num_filters)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids) # BxSxE
        embedded = embedded.permute(0, 2, 1) # BxExS

        # Áp dụng Conv1D
        features = torch.relu(self.conv(embedded))

        features = features.permute(0, 2, 1)
        features = self.norm(features)
        features = self.dropout(features)
        features = self.fc1(features)
        logits = self.fc2(features)
        

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1), labels)

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
                 num_layers=6, hidden_dim=1024, pad_idx=0):
        super().__init__(config)
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = PositionalEncoding(
            num_hiddens=embedding_dim, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        seq_length = input_ids.size(1)
        embedding = self.embedding(input_ids)
        outputs = self.position_embedding(embedding)

        outputs = self.transformer_encoder(outputs)

        logits = self.fc(outputs)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.permute(0, 2, 1), labels)
        return {"loss": loss, "logits": logits}