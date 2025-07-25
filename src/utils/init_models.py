import yaml
import os
from typing import Dict, Any

from ..models.encoder import ConvEncoder, PretrainedResNet18Encoder
from ..models.predictor import MLPHistoryPredictor
from ..models.decoder import ConvDecoder
from ..models.reward_predictor import MLPRewardPredictor
from ..models.vicreg import VICRegLoss
from ..data.data_utils import _initialize_environment

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


def init_encoder(config_path: str = None) -> ConvEncoder:
    """
    Initialize the ConvEncoder model from configuration.

    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized ConvEncoder model.
    """
    config = load_config(config_path)
    
    encoder_config = config['models']['encoder']
    latent_dim = config['latent_dim']

    encoder = ConvEncoder(
        latent_dim=latent_dim,
        input_channels=encoder_config.get('input_channels', 3),  # Default to RGB
        conv_channels=encoder_config['conv_channels'],
        activation=encoder_config['activation'],
        dropout_rate=encoder_config['dropout_rate'],
        use_pretrained_resnet=encoder_config.get('use_pretrained_resnet', True),
        image_size=encoder_config.get('image_size', 224)
    )
    
    return encoder


def init_conv_encoder(config_path: str = None) -> ConvEncoder:
    """
    Alias for init_encoder for backward compatibility with tests.
    """
    return init_encoder(config_path)



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

    data_config = config['data_and_patching']
    sequence_length = data_config['sequence_length']
    
    # Extract relevant configuration parameters
    predictor_config = config['models']['predictor']
    latent_dim = config['latent_dim']

    predictor = MLPHistoryPredictor(
        frames_per_clip=sequence_length,
        latent_dim=latent_dim,
        num_actions=num_actions,
        hidden_sizes=predictor_config['hidden_sizes'],
        activation=predictor_config['activation'],
        dropout_rate=predictor_config['dropout_rate']
    )

    return predictor


def init_decoder(config_path: str = None) -> ConvDecoder:
    """
    Initialize the HybridConvTransformerDecoder model from configuration.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Initialized HybridConvTransformerDecoder model.
    """
    config = load_config(config_path)
    
    # Extract relevant configuration parameters
    decoder_config = config['models']['decoder']
    
    decoder = ConvDecoder(
        latent_dim=config['latent_dim'],  # Use global latent_dim from config
        initial_size=decoder_config['initial_size'],
        conv_channels=decoder_config['conv_channels'],
        activation=decoder_config['activation'],
        dropout_rate=decoder_config['dropout_rate']
    )
    
    return decoder


def init_conv_decoder(config_path: str = None) -> ConvDecoder:
    """
    Alias for init_decoder for backward compatibility with tests.
    """
    return init_decoder(config_path)


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