import torch
import torch.nn as nn

class EnhancedLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        super(EnhancedLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        
        for lstm in self.lstm_layers:
            residual = x
            x, _ = lstm(x)
            x = x + residual
            x = self.layer_norm(x)
            x = self.dropout(x)
        
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        
        x = self.dropout(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
