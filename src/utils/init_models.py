import yaml
import os
from typing import Dict, Any

from ..models.encoder import VideoViT, ConvEncoder
from ..models.predictor import LatentDynamicsPredictor, MLPHistoryPredictor
from ..models.decoder import HybridConvTransformerDecoder, ConvDecoder
from ..models.reward_predictor import RewardPredictor
from ..models.vicreg import VICRegLoss
from ..data.data_utils import _initialize_environment
import torch

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


def init_encoder(config_path: str = None) -> VideoViT:
    """
    Initialize the VideoViT encoder model from configuration.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized VideoViT encoder model.
    """
    config = load_config(config_path)
    
    # Extract relevant configuration parameters
    data_config = config['data_and_patching']
    encoder_config = config['models']['encoder']
    embed_dim = config['embed_dim']
    
    #encoder = VideoViT(
    #    img_h=data_config['image_height'],
    #    img_w=data_config['image_width'],
    #    frames_per_clip=data_config['sequence_length'],
    #    patch_size_h=data_config['patch_size_h'],
    #    patch_size_w=data_config['patch_size_w'],
    #    embed_dim=embed_dim,
    #    mlp_ratio=encoder_config['mlp_ratio'],
    #    drop_rate=encoder_config['dropout'],
    #    attn_drop_rate=encoder_config['attention_dropout'],
    #    encoder_num_layers=encoder_config['num_layers'],
    #    encoder_num_heads=encoder_config['num_heads'],
    #    encoder_drop_path_rate=encoder_config['predictor_drop_path_rate']
    #)

    encoder = ConvEncoder()
    
    return encoder


def init_conv_encoder(config_path: str = None) -> ConvEncoder:
    """
    Initialize the ConvEncoder model.
    
    Args:
        config_path: Path to config.yaml file (not used, but kept for consistency).
    
    Returns:
        Initialized ConvEncoder model.
    """
    encoder = ConvEncoder()
    return encoder


def init_predictor(config_path: str = None) -> LatentDynamicsPredictor:
    """
    Initialize the LatentDynamicsPredictor model from configuration.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized LatentDynamicsPredictor model.
    """
    config = load_config(config_path)

    enc, render_mode = _initialize_environment(config)
    num_actions = enc.action_space.n
    print(f"Environment action space: {num_actions} actions")
    
    # Extract relevant configuration parameters
    predictor_config = config['models']['predictor']
    embed_dim = config['embed_dim']

    data_config = config['data_and_patching']

    #predictor = LatentDynamicsPredictor(
    #    frames_per_clip=data_config['sequence_length'],
    #    embed_dim=embed_dim,
    #    num_actions= num_actions,
    #    predictor_num_layers=predictor_config['num_layers'],
    #    predictor_num_heads=predictor_config['num_heads'],
    #    mlp_ratio=predictor_config['mlp_ratio'],
    #    drop_rate=predictor_config['dropout'],
    #    attn_drop_rate=predictor_config['attention_dropout'],
    #    predictor_drop_path_rate=predictor_config['predictor_drop_path_rate']
    #)

    predictor = MLPHistoryPredictor()

    return predictor


def init_decoder(config_path: str = None) -> HybridConvTransformerDecoder:
    """
    Initialize the HybridConvTransformerDecoder model from configuration.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized HybridConvTransformerDecoder model.
    """
    config = load_config(config_path)
    
    # Extract relevant configuration parameters
    data_config = config['data_and_patching']
    decoder_config = config['models']['decoder']
    embed_dim = config['embed_dim']
    
    #decoder = HybridConvTransformerDecoder(
    #    img_h=data_config['image_height'],
    #    img_w=data_config['image_width'],
    #    embed_dim=embed_dim,
    #    drop_rate=decoder_config['dropout'],
    #    attn_drop_rate=decoder_config['attention_dropout'],
    #    decoder_embed_dim=embed_dim,  # Use same embed_dim for decoder
    #    decoder_num_heads=decoder_config['num_heads'],
    #    decoder_drop_path_rate=decoder_config['decoder_drop_path_rate'],
    #    patch_size_h=data_config['patch_size_h'],
    #    patch_size_w=data_config['patch_size_w']
    #)

    decoder = ConvDecoder()
    
    return decoder


def init_reward_predictor(config_path: str = None) -> RewardPredictor:
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
    
    reward_predictor = RewardPredictor(
        embedding_dim=config['embed_dim'],
        internal_embedding_dim=reward_predictor_config['internal_embedding_dim'],
        num_heads=reward_predictor_config['num_heads'],
        num_attention_layers=reward_predictor_config.get('num_attention_layers', 1),
        mlp_hidden_layers=reward_predictor_config.get('mlp_hidden_layers', None),
        dropout=reward_predictor_config.get('dropout', 0.1),
        attention_dropout=reward_predictor_config.get('attention_dropout', 0.1),
        use_layer_norm=reward_predictor_config.get('use_layer_norm', True),
        activation=reward_predictor_config.get('activation', 'relu')
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