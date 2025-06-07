# models/__init__.py
from .vit import ViT
from .encoder_decoder import StandardEncoderDecoder
from .jepa import JEPA
from .cnn import CNNEncoder
from .mlp import MLPEncoder  # Add this line

__all__ = [
    "ViT",
    "StandardEncoderDecoder",
    "JEPA",
    "CNNEncoder",
    "MLPEncoder"  # Add this line
]
