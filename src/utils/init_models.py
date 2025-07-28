import yaml
import os
from typing import Dict, Any

from ..models.encoder import Encoder
from ..models.predictor import MLPHistoryPredictor
from ..models.decoder import Decoder
from ..models.reward_predictor import MLPRewardPredictor
from ..models.vicreg import VICRegLoss
from ..data.data_utils import _initialize_environment
from transformers import ViTMAEConfig, ViTMAEForPreTraining, Trainer, TrainingArguments


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


def init_encoder(config_path: str = None) -> Encoder:
    """
    Initialize the ConvEncoder model from configuration.

    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized ConvEncoder model.
    """
    config = load_config(config_path)
    latent_dim = config['latent_dim']

    encoder = Encoder(
        latent_dim=latent_dim,
    )
    
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
    config = load_config(config_path)
    
    decoder = Decoder(
        latent_dim=config['latent_dim'],  # Use global latent_dim from config
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

def init_vit_mae(config_path: str = None):
    """
    Initialize the MAE model from configuration.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized MAE model.
    """
 
    config = ViTMAEConfig(
            # encoder (ViT‐Base defaults would be hidden_size=768)
            hidden_size=64,                # your “latent_dim”
            num_hidden_layers=2,           # same depth as ViT‐Base
            num_attention_heads=8,          # 512/8 = 64‑dim per head
            intermediate_size=128,         # 4× hidden_size
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-12,

            # image & patch sizing
            image_size=84,                  # match your 84×84 frames
            patch_size=7,                   # 84/7 = 12 patches per side
            num_channels=4,                 # stack of 4 grayscale frames

            # decoder (lightweight MAE decoder)
            decoder_hidden_size=64,        # mirror encoder latent
            decoder_num_hidden_layers=2,    # shallow “head”
            decoder_num_attention_heads=16, # head_dim = 512/16 = 32
            decoder_intermediate_size=128,

            # masking & loss
            mask_ratio=0.35,                # same 75% masking
            norm_pix_loss=False             # or True if you want pixel‐norm loss
        )

    model = ViTMAEForPreTraining(config)

    return model