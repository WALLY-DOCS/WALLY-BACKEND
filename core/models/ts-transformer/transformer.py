import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        # Precompute positional encodings
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encodings to the input
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super(TimeSeriesTransformer, self).__init__()
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, output_dim)
        )
        
        # Initialize parameters
        self._reset_parameters()

        # Model dimension
        self.d_model = d_model
        
    def _reset_parameters(self):
        # Default initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project input to model dimension
        x = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encodings
        x = self.pos_encoder(x)
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x, src_mask)
        
        # Apply the output projection (take the last time step)
        return self.output_layer(x[:, -1, :])

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        """Generate a causal mask for Transformer attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

class TimeSeriesTransformerTrainer:
    def __init__(self, model, learning_rate=1e-4, scaler=None):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.MSELoss()
        self.scaler = scaler  # For mixed precision training
    
    def train_step(self, batch: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Create causal mask
        mask = self.model.generate_square_subsequent_mask(batch.size(1)).to(batch.device)
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            output = self.model(batch, mask)
            loss = self.criterion(output, targets)
        
        if self.scaler is not None:
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch, targets in val_loader:
                mask = self.model.generate_square_subsequent_mask(batch.size(1)).to(batch.device)
                output = self.model(batch, mask)
                loss = self.criterion(output, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.scheduler.step(avg_loss)
        return avg_loss
