"""
Variational Autoencoder (VAE) Model for DNA Sequence Analysis

This module implements a VAE using 1D convolutional layers for encoding and decoding
DNA sequences in one-hot encoded format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for DNA sequence analysis.
    
    The model uses 1D convolutional layers to encode DNA sequences into a latent space
    and decode them back to reconstruct the original sequences.
    """
    
    def __init__(self, 
                 sequence_length: int = 200,
                 n_channels: int = 4,  # A, T, C, G
                 latent_dim: int = 32,
                 hidden_dims: list = [64, 128, 256]):
        """
        Initialize the VAE model.
        
        Args:
            sequence_length (int): Length of input DNA sequences
            n_channels (int): Number of channels (nucleotides: A, T, C, G)
            latent_dim (int): Dimension of the latent space
            hidden_dims (list): List of hidden dimensions for encoder/decoder
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder input layer
        self.fc_decoder = nn.Linear(latent_dim, hidden_dims[-1])
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        logger.info(f"VAE initialized with latent_dim={latent_dim}, sequence_length={sequence_length}")
    
    def _build_encoder(self) -> nn.Module:
        """Build the encoder network using 1D convolutional layers."""
        layers = []
        in_channels = self.n_channels
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_channels = hidden_dim
        
        # Global average pooling
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder network using 1D transposed convolutional layers."""
        layers = []
        
        # Calculate the size after encoding
        encoded_size = self.sequence_length
        for _ in self.hidden_dims:
            encoded_size = (encoded_size + 1) // 2  # Due to stride=2
        
        # Reshape and upsample
        layers.append(nn.Unflatten(1, (self.hidden_dims[-1], 1)))
        
        # Transposed convolutions
        hidden_dims_reversed = self.hidden_dims[::-1]
        for i in range(len(hidden_dims_reversed) - 1):
            layers.extend([
                nn.ConvTranspose1d(hidden_dims_reversed[i], 
                                  hidden_dims_reversed[i + 1], 
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(hidden_dims_reversed[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
        
        # Final layer to output channels
        layers.extend([
            nn.ConvTranspose1d(hidden_dims_reversed[-1], self.n_channels, 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Softmax(dim=1)  # Softmax over nucleotide channels
        ])
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequences into latent space.
        
        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, n_channels, sequence_length)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mu, log_var) of the latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            log_var (torch.Tensor): Log variance of the latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector back to sequence space.
        
        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Reconstructed sequences of shape (batch_size, n_channels, sequence_length)
        """
        h = self.fc_decoder(z)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, n_channels, sequence_length)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (reconstructed_x, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, log_var
    
    def get_latent_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representations without sampling (deterministic).
        
        Args:
            x (torch.Tensor): Input sequences
            
        Returns:
            torch.Tensor: Mean latent representations
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu

def vae_loss(reconstructed_x: torch.Tensor, 
             x: torch.Tensor, 
             mu: torch.Tensor, 
             log_var: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
    """
    Calculate the VAE loss (reconstruction loss + KL divergence).
    
    Args:
        reconstructed_x (torch.Tensor): Reconstructed sequences
        x (torch.Tensor): Original sequences
        mu (torch.Tensor): Mean of latent distribution
        log_var (torch.Tensor): Log variance of latent distribution
        beta (float): Weight for KL divergence term
        
    Returns:
        torch.Tensor: Total VAE loss
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss

def create_vae_model(sequence_length: int = 200, 
                    latent_dim: int = 32,
                    device: str = 'cpu') -> VariationalAutoencoder:
    """
    Create and initialize a VAE model.
    
    Args:
        sequence_length (int): Length of DNA sequences
        latent_dim (int): Dimension of latent space
        device (str): Device to run the model on
        
    Returns:
        VariationalAutoencoder: Initialized VAE model
    """
    model = VariationalAutoencoder(
        sequence_length=sequence_length,
        latent_dim=latent_dim
    )
    
    model = model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    logger.info(f"VAE model created and moved to {device}")
    return model

if __name__ == "__main__":
    # Test the VAE model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_vae_model(sequence_length=200, latent_dim=32, device=device)
    
    # Test with random input
    batch_size = 4
    test_input = torch.randn(batch_size, 4, 200).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    reconstructed, mu, log_var = model(test_input)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log var shape: {log_var.shape}")
    
    # Calculate loss
    loss = vae_loss(reconstructed, test_input, mu, log_var)
    print(f"VAE Loss: {loss.item():.4f}")
    
    # Test latent representations
    latent_repr = model.get_latent_representations(test_input)
    print(f"Latent representations shape: {latent_repr.shape}")
    
    print("VAE model test completed successfully!")
