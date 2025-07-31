import yaml
import os
from typing import Dict, Any

from ..models.encoder import VisionTransformerEncoder
from ..models.predictor import MLPHistoryPredictor
from ..models.decoder import Decoder
from ..models.reward_predictor import MLPRewardPredictor
from ..models.vicreg import VICRegLoss
from ..data.data_utils import _initialize_environment
from dataclasses import dataclass

@dataclass
class ViTConfig:
    # encoder
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 512
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12

    # image & patch
    image_size: int = 84
    patch_size: int = 7
    num_channels: int = 4

    # decoder (ViT‑MAE)
    decoder_hidden_size: int = 128
    decoder_num_hidden_layers: int = 4
    decoder_num_attention_heads: int = 16
    decoder_intermediate_size: int = 512

    # masking & loss
    mask_ratio: float = 0.35


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from yaml file.
    
    Args:
        config_path: Path to config.yaml file. If None, looks for config.yaml in project root.
    
    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        # Default to config.yaml in project root (two levels up from utils)
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def init_encoder(config_path: str = None) -> VisionTransformerEncoder:
    """
    Initialize the ConvEncoder model from configuration.

    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized ConvEncoder model.
    """
    config = ViTConfig() 
    encoder = VisionTransformerEncoder(config)

    return encoder



def init_predictor(config_path: str = None) -> MLPHistoryPredictor:
    """
    Initialize the LatentDynamicsPredictor model from configuration.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized LatentDynamicsPredictor model.
    """
    config = load_config(config_path)

    env, render_mode = _initialize_environment(config)
    num_actions = env.action_space.n
    
    # Extract relevant configuration parameters
    predictor_config = config['models']['predictor']
    latent_dim = config['latent_dim']

    predictor = MLPHistoryPredictor(
        latent_dim=latent_dim,
        num_actions=num_actions,
        hidden_sizes=predictor_config['hidden_sizes'],
        activation=predictor_config['activation'],
        dropout_rate=predictor_config['dropout_rate']
    )

    return predictor


def init_decoder(config_path: str = None) -> Decoder:
    """
    Initialize the HybridConvTransformerDecoder model from configuration.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized HybridConvTransformerDecoder model.
    """
    config = ViTConfig()  # Use ViTConfig for decoder as well
    
    decoder = Decoder(
        config
    )
    
    return decoder


def init_reward_predictor(config_path: str = None) -> MLPRewardPredictor:
    """
    Initialize the RewardPredictor model from configuration.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized RewardPredictor model.
    """
    config = load_config(config_path)
    
    # Extract relevant configuration parameters
    reward_predictor_config = config['models']['reward_predictor']
    
    reward_predictor = MLPRewardPredictor(
        latent_dim=config['latent_dim'],
        sequence_length=config['data_and_patching']['sequence_length'],
        hidden_dims=reward_predictor_config.get('hidden_sizes'),
        dropout=reward_predictor_config.get('dropout')
    )
    
    return reward_predictor 

def init_vicreg(config_path: str = None) -> VICRegLoss:
    config = load_config(config_path)
    vicreg_config = config['models']['vicreg']
    use_vicreg = vicreg_config.get('active')
    if not use_vicreg:
        return None
    vicreg = VICRegLoss(
        sim_coeff=vicreg_config.get('lambda_i'),
        cov_coeff=vicreg_config.get('lambda_s'),
        std_coeff=vicreg_config.get('lambda_v'),
        proj_hidden_dim=vicreg_config.get('proj_hidden_dim'),
        proj_output_dim=vicreg_config.get('proj_output_dim'),
        representation_dim=config['embed_dim'],
    )
    return vicreg