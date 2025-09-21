"""Autoencoder models for ImageNet-64 fragment reconstruction."""

from typing import Any

import torch
from torch import nn

from .modules import ConvBlock, DeconvBlock


class ConvAutoencoder(nn.Module):
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        encoder_channels: list[int] | None = None,
        input_size: int = 16
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        if encoder_channels is None:
            encoder_channels = [32, 64]
        
        self.encoder_channels = encoder_channels
        self.num_layers = len(encoder_channels)
        
        self.final_spatial_size = input_size // (2 ** self.num_layers)
        if self.final_spatial_size < 1:
            raise ValueError(f"Too many layers for input size {input_size}. Maximum layers: {input_size.bit_length() - 1}")
        
        encoder_layers = []
        in_channels = input_channels
        for i, out_channels in enumerate(encoder_channels):
            encoder_layers.append(self._make_conv_block(in_channels, out_channels, stride=2))
            in_channels = out_channels
        self.encoder = nn.ModuleList(encoder_layers)
        
        encoder_output_size = encoder_channels[-1] * (self.final_spatial_size ** 2)
        self.encode_fc = nn.Linear(encoder_output_size, latent_dim)
        self.decode_fc = nn.Linear(latent_dim, encoder_output_size)
        
        decoder_layers = []
        decoder_channels = list(reversed(encoder_channels))
        for i in range(len(decoder_channels) - 1):
            in_channels = decoder_channels[i]
            out_channels = decoder_channels[i + 1]
            decoder_layers.append(self._make_deconv_block(in_channels, out_channels, stride=2))
        decoder_layers.append(
            self._make_deconv_block(decoder_channels[-1], input_channels, stride=2, final_layer=True)
        )
        self.decoder = nn.ModuleList(decoder_layers)
        
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
        if stride == 1:
            return nn.Sequential(
                ConvBlock(in_channels, out_channels),
                ConvBlock(out_channels, out_channels),
            )
        else:
            return nn.Sequential(
                ConvBlock(in_channels, out_channels, stride=stride),
                ConvBlock(out_channels, out_channels),
            )
    
    def _make_deconv_block(self, in_channels: int, out_channels: int, stride: int = 1, final_layer: bool = False) -> nn.Module:
        if not final_layer:
            return nn.Sequential(
                DeconvBlock(in_channels, out_channels, stride=stride),
                ConvBlock(out_channels, out_channels),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.Sigmoid(),
            )
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        latent: torch.Tensor = self.encode_fc(x_flat)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.decode_fc(latent)
        batch_size = x.size(0)
        x_reshaped = x.view(
            batch_size,
            self.encoder_channels[-1],
            self.final_spatial_size,
            self.final_spatial_size,
        )
        
        for layer in self.decoder:
            x_reshaped = layer(x_reshaped)
        
        return x_reshaped
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class LinearAutoencoder(nn.Module):
    """Multi-layer linear autoencoder with same depth as convolutional version."""
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
        input_size: int = 16
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.input_dim = input_channels * input_size * input_size
        
        if hidden_dims is None:
            # Match the conv AE structure: input -> 512 -> 256 -> latent
            hidden_dims = [512, 256]
        
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        decoder_layers.extend([
            nn.Linear(in_dim, self.input_dim),
            nn.Sigmoid()
        ])
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        encoded: torch.Tensor = self.encoder(x_flat)
        return encoded
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.decoder(latent)
        batch_size = x.size(0)
        output: torch.Tensor = x.view(batch_size, self.input_channels, self.input_size, self.input_size)
        return output
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class PCAAutoencoder(nn.Module):
    """Single linear layer autoencoder (PCA-like)."""
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        input_size: int = 16
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.input_dim = input_channels * input_size * input_size
        
        # Single linear layer for encoding and decoding (like PCA)
        self.encoder = nn.Linear(self.input_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.input_dim),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        encoded: torch.Tensor = self.encoder(x_flat)
        return encoded
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.decoder(latent)
        batch_size = x.size(0)
        output: torch.Tensor = x.view(batch_size, self.input_channels, self.input_size, self.input_size)
        return output
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class SupervisedClassifier(nn.Module):
    """Supervised classifier using fragment labels instead of reconstruction."""
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        num_classes: int = 1000,  # Number of source images
        hidden_dims: list[int] | None = None,
        input_size: int = 16
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_dim = input_channels * input_size * input_size
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Feature extractor (encoder)
        encoder_layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Classifier head
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        encoded: torch.Tensor = self.encoder(x_flat)
        return encoded
    
    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(latent)
        return logits
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        logits = self.classify(latent)
        return logits, latent


def create_autoencoder(
    input_channels: int = 3,
    latent_dim: int = 128,
    encoder_channels: list[int] | None = None,
    input_size: int = 16
) -> ConvAutoencoder:
    return ConvAutoencoder(input_channels, latent_dim, encoder_channels, input_size)


def create_linear_autoencoder(
    input_channels: int = 3,
    latent_dim: int = 128,
    hidden_dims: list[int] | None = None,
    input_size: int = 16
) -> LinearAutoencoder:
    return LinearAutoencoder(input_channels, latent_dim, hidden_dims, input_size)


def create_pca_autoencoder(
    input_channels: int = 3,
    latent_dim: int = 128,
    input_size: int = 16
) -> PCAAutoencoder:
    return PCAAutoencoder(input_channels, latent_dim, input_size)


def create_supervised_classifier(
    input_channels: int = 3,
    latent_dim: int = 128,
    num_classes: int = 1000,
    hidden_dims: list[int] | None = None,
    input_size: int = 16
) -> SupervisedClassifier:
    return SupervisedClassifier(input_channels, latent_dim, num_classes, hidden_dims, input_size)


def count_parameters(model: Any) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: Any object that has a parameters() method
        
    Returns:
        int: Total number of trainable parameters
    """
    if not hasattr(model, 'parameters') or not callable(model.parameters):
        raise ValueError("Model must have a callable 'parameters' method")
        
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test all model types
    x = torch.randn(4, 3, 16, 16)
    
    # 1. Convolutional Autoencoder
    conv_model = create_autoencoder()
    print(f"Convolutional Autoencoder: {count_parameters(conv_model):,} parameters")
    reconstructed, latent = conv_model(x)
    print(f"Conv AE - Input: {x.shape}, Latent: {latent.shape}, Reconstructed: {reconstructed.shape}")
    
    # 2. Linear Autoencoder
    linear_model = create_linear_autoencoder()
    print(f"Linear Autoencoder: {count_parameters(linear_model):,} parameters")
    reconstructed, latent = linear_model(x)
    print(f"Linear AE - Input: {x.shape}, Latent: {latent.shape}, Reconstructed: {reconstructed.shape}")
    
    # 3. PCA Autoencoder
    pca_model = create_pca_autoencoder()
    print(f"PCA Autoencoder: {count_parameters(pca_model):,} parameters")
    reconstructed, latent = pca_model(x)
    print(f"PCA AE - Input: {x.shape}, Latent: {latent.shape}, Reconstructed: {reconstructed.shape}")
    
    # 4. Supervised Classifier
    supervised_model = create_supervised_classifier(num_classes=100)
    print(f"Supervised Classifier: {count_parameters(supervised_model):,} parameters")
    logits, latent = supervised_model(x)
    print(f"Supervised - Input: {x.shape}, Latent: {latent.shape}, Logits: {logits.shape}")
