# Contents for src/model_setup.py
import torch
from src.models.encoder_decoder import StandardEncoderDecoder
from src.models.jepa import JEPA
from src.models.mlp import RewardPredictorMLP
from src.models.jepa_state_decoder import JEPAStateDecoder # Added import
from src.models.encoder_decoder_jepa_style import EncoderDecoderJEPAStyle

def initialize_models(config, action_dim, device, image_h_w, input_channels):
    models = {}

    # Load Reward Predictor Configurations
    reward_pred_config = config.get('models', {}).get('reward_predictors', {})
    enc_dec_mlp_config = reward_pred_config.get('encoder_decoder_reward_mlp', {})
    jepa_mlp_config = reward_pred_config.get('jepa_reward_mlp', {})

    # Encoder configuration
    models_config = config.get('models', {})
    encoder_config = models_config.get('encoder', {})
    encoder_type = encoder_config.get('type', 'vit')
    all_encoder_params_from_config = encoder_config.get('params', {})
    specific_encoder_params = all_encoder_params_from_config.get(encoder_type, {})
    if specific_encoder_params is None:
        specific_encoder_params = {}

    global_patch_size = models_config.get('shared_patch_size', 16)
    shared_latent_dim = models_config.get('shared_latent_dim', 128)

    if encoder_type == 'vit':
        if 'patch_size' not in specific_encoder_params or specific_encoder_params['patch_size'] is None:
            specific_encoder_params['patch_size'] = global_patch_size
        # Use the specific encoder params from the new config structure
        specific_encoder_params.setdefault('depth', specific_encoder_params.get('depth', 6))
        specific_encoder_params.setdefault('heads', specific_encoder_params.get('heads', 8))
        specific_encoder_params.setdefault('mlp_dim', specific_encoder_params.get('mlp_dim', 1024))
        specific_encoder_params.setdefault('pool', specific_encoder_params.get('pool', 'cls'))
        specific_encoder_params.setdefault('dropout', specific_encoder_params.get('dropout', 0.0))
        specific_encoder_params.setdefault('emb_dropout', specific_encoder_params.get('emb_dropout', 0.0))

    print(f"Initializing Standard Encoder-Decoder Model with {encoder_type.upper()} encoder...")
    std_enc_dec_config = models_config.get('standard_encoder_decoder', {})
    std_enc_dec = StandardEncoderDecoder(
        image_size=image_h_w,
        patch_size=global_patch_size,
        input_channels=input_channels,
        action_dim=action_dim,
        action_emb_dim=std_enc_dec_config.get('action_emb_dim', shared_latent_dim),
        latent_dim=shared_latent_dim,
        decoder_dim=std_enc_dec_config.get('decoder_dim', 128),
        decoder_depth=std_enc_dec_config.get('decoder_depth', 3),
        decoder_heads=std_enc_dec_config.get('decoder_heads', 6),
        decoder_mlp_dim=std_enc_dec_config.get('decoder_mlp_dim', 256),
        output_channels=input_channels,
        output_image_size=image_h_w if isinstance(image_h_w, tuple) else (image_h_w, image_h_w),
        decoder_dropout=std_enc_dec_config.get('decoder_dropout', 0.0),
        encoder_type=encoder_type,
        encoder_params=specific_encoder_params,
        decoder_patch_size=std_enc_dec_config.get('decoder_patch_size', global_patch_size)
    ).to(device)
    models['std_enc_dec'] = std_enc_dec

    print(f"Initializing JEPA Model with {encoder_type.upper()} encoder...")
    jepa_config = models_config.get('jepa', {})
    jepa_model = JEPA(
        image_size=image_h_w,
        patch_size=global_patch_size,
        input_channels=input_channels,
        action_dim=action_dim,
        action_emb_dim=std_enc_dec_config.get('action_emb_dim', shared_latent_dim),
        latent_dim=shared_latent_dim,
        predictor_hidden_dim=jepa_config.get('predictor_hidden_dim', 256),
        predictor_output_dim=shared_latent_dim, # JEPA predictor output dim is same as latent_dim
        ema_decay=jepa_config.get('ema_decay', 0.996),
        encoder_type=encoder_type,
        encoder_params=specific_encoder_params,
        predictor_dropout_rate=jepa_config.get('predictor_dropout_rate', 0.0)
    ).to(device)
    models['jepa'] = jepa_model

    reward_mlp_enc_dec = None
    if enc_dec_mlp_config.get('enabled', False):
        print("Initializing Reward MLP for Encoder-Decoder...")
        # Ensure image_h_w is treated as a single dimension if it's an int for area calculation
        img_h = image_h_w[0] if isinstance(image_h_w, tuple) else image_h_w
        img_w = image_h_w[1] if isinstance(image_h_w, tuple) else image_h_w
        if enc_dec_mlp_config.get('input_type') == "flatten":
            input_dim_enc_dec = input_channels * img_h * img_w
        else:
            print(f"Warning: encoder_decoder_reward_mlp input_type is '{enc_dec_mlp_config.get('input_type')}'. Defaulting to flattened image dim.")
            input_dim_enc_dec = input_channels * img_h * img_w

        reward_mlp_enc_dec = RewardPredictorMLP(
            input_dim=input_dim_enc_dec,
            hidden_dims=enc_dec_mlp_config.get('hidden_dims', [128, 64]),
            activation_fn_str=enc_dec_mlp_config.get('activation', 'relu'),
            use_batch_norm=enc_dec_mlp_config.get('use_batch_norm', False),
            dropout_rate=enc_dec_mlp_config.get('dropout_rate', 0.0) # Added
        ).to(device)
        print(f"Encoder-Decoder Reward MLP: {reward_mlp_enc_dec}")
    models['reward_mlp_enc_dec'] = reward_mlp_enc_dec

    reward_mlp_jepa = None
    if jepa_mlp_config.get('enabled', False):
        print("Initializing Reward MLP for JEPA...")
        input_dim_jepa = shared_latent_dim # JEPA's reward MLP uses latent_dim
        reward_mlp_jepa = RewardPredictorMLP(
            input_dim=input_dim_jepa,
            hidden_dims=jepa_mlp_config.get('hidden_dims', [128, 64]),
            activation_fn_str=jepa_mlp_config.get('activation', 'relu'),
            use_batch_norm=jepa_mlp_config.get('use_batch_norm', False),
            dropout_rate=jepa_mlp_config.get('dropout_rate', 0.0) # Added
        ).to(device)
        print(f"JEPA Reward MLP: {reward_mlp_jepa}")
    models['reward_mlp_jepa'] = reward_mlp_jepa

    # Initialize JEPA State Decoder
    jepa_decoder_config = jepa_config.get('decoder_training', {})
    if jepa_decoder_config.get('enabled', False):
        print("Initializing JEPA State Decoder...")

        # Ensure image_h_w is a tuple for JEPAStateDecoder
        current_image_h_w = image_h_w if isinstance(image_h_w, tuple) else (image_h_w, image_h_w)

        jepa_decoder = JEPAStateDecoder(
            input_latent_dim=shared_latent_dim, # JEPA's predictor output dim
            decoder_dim=std_enc_dec_config.get('decoder_dim', 128),
            decoder_depth=std_enc_dec_config.get('decoder_depth', 3),
            decoder_heads=std_enc_dec_config.get('decoder_heads', 4),
            decoder_mlp_dim=std_enc_dec_config.get('decoder_mlp_dim', 256),
            output_channels=input_channels,
            output_image_size=current_image_h_w,
            decoder_dropout=std_enc_dec_config.get('decoder_dropout', 0.0),
            decoder_patch_size=std_enc_dec_config.get('decoder_patch_size', global_patch_size) # Default to global patch_size if specific not found
        ).to(device)
        models['jepa_decoder'] = jepa_decoder
        print(f"JEPA State Decoder: {jepa_decoder}")
    else:
        models['jepa_decoder'] = None
        print("JEPA State Decoder is disabled in the configuration.")

    # Encoder-Decoder JEPA-Style Model (for fair JEPA comparison)
    print(f"Initializing Encoder-Decoder JEPA-Style Model with {encoder_type.upper()} encoder...")
    enc_dec_jepa_style_config = models_config.get('enc_dec_jepa_style', {})
    enc_dec_jepa_style = EncoderDecoderJEPAStyle(
        image_size=image_h_w,
        patch_size=global_patch_size,
        input_channels=input_channels,
        action_dim=action_dim,
        action_emb_dim=std_enc_dec_config.get('action_emb_dim', shared_latent_dim),
        latent_dim=shared_latent_dim,
        predictor_hidden_dim=enc_dec_jepa_style_config.get('predictor_hidden_dim', 256),
        predictor_output_dim=enc_dec_jepa_style_config.get('predictor_output_dim', std_enc_dec_config.get('decoder_dim', 128)),
        predictor_dropout_rate=enc_dec_jepa_style_config.get('predictor_dropout_rate', 0.0),
        decoder_dim=std_enc_dec_config.get('decoder_dim', 128),
        decoder_depth=std_enc_dec_config.get('decoder_depth', 3),
        decoder_heads=std_enc_dec_config.get('decoder_heads', 6),
        decoder_mlp_dim=std_enc_dec_config.get('decoder_mlp_dim', 256),
        output_channels=input_channels,
        output_image_size=image_h_w if isinstance(image_h_w, tuple) else (image_h_w, image_h_w),
        decoder_dropout=std_enc_dec_config.get('decoder_dropout', 0.0),
        encoder_type=encoder_type,
        encoder_params=specific_encoder_params,
        decoder_patch_size=std_enc_dec_config.get('decoder_patch_size', global_patch_size)
    ).to(device)
    models['enc_dec_jepa_style'] = enc_dec_jepa_style

    return models
