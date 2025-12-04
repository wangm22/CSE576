import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    Uses sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model (embedding size)
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    Simple Transformer model for stock market return prediction.
    """
    
    def __init__(self, 
                 input_dim,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_seq_len=100):
        """
        Args:
            input_dim: Number of input features (94 in our case)
            d_model: Dimension of the model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(SimpleTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection: projects input features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: project to single value (return prediction)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass of the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               e.g., (batch_size, 30, 94)
        
        Returns:
            Predicted returns of shape (batch_size, 1)
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the last timestep's output for prediction
        x = x[:, -1, :]
        
        # Output projection
        out = self.output_projection(x)
        
        return out.squeeze(-1)
    
    def get_attention_weights(self, x):
        """
        Get attention weights for visualization (optional).
        Note: This requires modifying the transformer encoder to return attention weights.
        """
        # This is a placeholder - implement if you need attention visualization
        pass


def create_model(input_dim=14,
                d_model=128,
                nhead=8,
                num_layers=4,
                dim_feedforward=512,
                dropout=0.1):
    """
    Factory function to create a SimpleTransformer model.

    Args:
        input_dim: Number of input features (default: 14)
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate

    Returns:
        SimpleTransformer model
    """
    model = SimpleTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("="*80)
    print("TESTING SIMPLE TRANSFORMER MODEL")
    print("="*80)
    
    # Model hyperparameters
    input_dim = 14
    seq_len = 30
    batch_size = 32

    # Create model
    print("\nCreating model...")
    model = create_model(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Print model architecture
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(model)
    
    # Count parameters
    n_params = count_parameters(model)
    print("\n" + "="*80)
    print(f"Total trainable parameters: {n_params:,}")
    print("="*80)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions (first 5): {output[:5].numpy()}")
    
    print("\n" + "="*80)
    print("MODEL TEST COMPLETE ✓")
    print("="*80)
    
    # Print model summary
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(f"Input dimension: {input_dim} features (11 base + 3 lagged)")
    print(f"Sequence length: {seq_len} timesteps")
    print(f"Model dimension: 128")
    print(f"Attention heads: 8")
    print(f"Encoder layers: 4")
    print(f"Feedforward dimension: 512")
    print(f"Total parameters: {n_params:,}")
    print(f"Output: Single return prediction per sample")