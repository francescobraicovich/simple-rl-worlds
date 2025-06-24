# Contents for src/model_setup.py
import torch
from src.models.encoder_decoder import StandardEncoderDecoder
from src.models.jepa import JEPA
from src.models.mlp import RewardPredictorMLP
from src.models.jepa_state_decoder import JEPAStateDecoder # Added import
from src.models.encoder_decoder_jepa_style import EncoderDecoderJEPAStyle
from copy import deepcopy
from src.models.larp_mlp import LookAheadRewardPredictorMLP # Added for LARP
from src.utils.larp_utils import calculate_larp_input_dim_enc_dec, calculate_larp_input_dim_jepa # Added for LARP

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
    # Read the variant from the config
    encoder_decoder_variant = std_enc_dec_config.get('variant', 'standard')

    jepa_config = models_config.get('jepa', {}) # Used by both variants for some params
    jepa_state_decoder_arch_config = models_config.get('jepa_state_decoder_arch', {}) # Used by jepa_style

    if encoder_decoder_variant == 'jepa_style':
        print(f"Selected Encoder-Decoder Variant: JEPA-Style with {encoder_type.upper()} encoder.")
        # Parameters for EncoderDecoderJEPAStyle, sourcing from various config sections as per comments in config.yaml
        # Encoder params are already prepared (encoder_type, specific_encoder_params, shared_latent_dim)
        # Action embedding dim from standard_encoder_decoder config (convention)
        action_emb_dim_jepa_style = std_enc_dec_config.get('action_emb_dim', shared_latent_dim)
        
        # Predictor params from jepa config
        predictor_hidden_dim_jepa_style = jepa_config.get('predictor_hidden_dim', 256)
        predictor_dropout_rate_jepa_style = jepa_config.get('predictor_dropout_rate', 0.0)
        # Predictor output_dim for enc_dec_jepa_style is the input_latent_dim of its internal JEPAStateDecoder
        # This comes from jepa_state_decoder_arch.input_latent_dim
        predictor_output_dim_jepa_style = jepa_state_decoder_arch_config.get('input_latent_dim', shared_latent_dim)

        # Internal JEPAStateDecoder architectural params from jepa_state_decoder_arch config
        # Note: input_latent_dim for this internal decoder is predictor_output_dim_jepa_style
        jepa_decoder_dim_internal = jepa_state_decoder_arch_config.get('decoder_dim', 128)
        jepa_decoder_depth_internal = jepa_state_decoder_arch_config.get('decoder_depth', 4)
        jepa_decoder_heads_internal = jepa_state_decoder_arch_config.get('decoder_heads', 4)
        jepa_decoder_mlp_dim_internal = jepa_state_decoder_arch_config.get('decoder_mlp_dim', 512)
        jepa_decoder_dropout_internal = jepa_state_decoder_arch_config.get('decoder_dropout', 0.2)
        # Decoder patch size: from jepa_state_decoder_arch, fallback to global_patch_size
        jepa_decoder_patch_size_internal = jepa_state_decoder_arch_config.get('decoder_patch_size', global_patch_size)

        # Output channels and image size for the final output (from environment config, passed as args)
        # output_channels_internal = input_channels (function arg)
        # output_image_size_internal = image_h_w (function arg)

        std_enc_dec = EncoderDecoderJEPAStyle(
            image_size=image_h_w,
            patch_size=global_patch_size, # For ViT encoder and default for decoder patch size
            input_channels=input_channels,
            action_dim=action_dim,
            action_emb_dim=action_emb_dim_jepa_style,
            latent_dim=shared_latent_dim, # Encoder output latent dim

            predictor_hidden_dim=predictor_hidden_dim_jepa_style,
            predictor_output_dim=predictor_output_dim_jepa_style, # This is input to internal JEPAStateDecoder
            predictor_dropout_rate=predictor_dropout_rate_jepa_style,

            # Internal JEPAStateDecoder architecture params
            jepa_decoder_dim=jepa_decoder_dim_internal,
            jepa_decoder_depth=jepa_decoder_depth_internal,
            jepa_decoder_heads=jepa_decoder_heads_internal,
            jepa_decoder_mlp_dim=jepa_decoder_mlp_dim_internal,
            jepa_decoder_dropout=jepa_decoder_dropout_internal,
            jepa_decoder_patch_size=jepa_decoder_patch_size_internal,

            output_channels=input_channels, # Final output channels
            output_image_size=image_h_w if isinstance(image_h_w, tuple) else (image_h_w, image_h_w), # Final output image size

            encoder_type=encoder_type,
            encoder_params=specific_encoder_params
        ).to(device)

    elif encoder_decoder_variant == 'standard':
        print(f"Selected Encoder-Decoder Variant: Standard with {encoder_type.upper()} encoder.")
        std_enc_dec = StandardEncoderDecoder(
            image_size=image_h_w,
            patch_size=global_patch_size, # For ViT encoder and default for decoder patch size
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
    else:
        raise ValueError(f"Unknown encoder_decoder_variant: {encoder_decoder_variant} in config.yaml")

    models['std_enc_dec'] = std_enc_dec # Store the selected model

    print(f"Initializing JEPA Model with {encoder_type.upper()} encoder...")
    # jepa_config already fetched
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
        if enc_dec_mlp_config.get('input_type') == "flatten":
            input_dim_enc_dec = shared_latent_dim + action_emb_dim_jepa_style # Flattened input: latent_dim + action_dim
        else:
            print(f"Warning: encoder_decoder_reward_mlp input_type is '{enc_dec_mlp_config.get('input_type')}'. Defaulting to flattened image dim.")
            input_dim_enc_dec = shared_latent_dim + action_emb_dim_jepa_style
        
        reward_mlp_enc_dec = RewardPredictorMLP(
            input_dim=input_dim_enc_dec,
            hidden_dims=enc_dec_mlp_config.get('hidden_dims', [128, 64]),
            activation_fn_str=enc_dec_mlp_config.get('activation', 'relu'),
            use_batch_norm=enc_dec_mlp_config.get('use_batch_norm', False),
            dropout_rate=enc_dec_mlp_config.get('dropout_rate', 0.0) # Added
        ).to(device)
    models['reward_mlp_enc_dec'] = reward_mlp_enc_dec

    reward_mlp_jepa = None
    if jepa_mlp_config.get('enabled', False):
        print("Initializing Reward MLP for JEPA...")
        # copy the reward predictor for encoder-decoder
        reward_mlp_jepa = deepcopy(reward_mlp_enc_dec)

    models['reward_mlp_jepa'] = reward_mlp_jepa

    # Initialize JEPA State Decoder
    jepa_decoder_config = jepa_config.get('decoder_training', {})
    if jepa_decoder_config.get('enabled', False):

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
    else:
        models['jepa_decoder'] = None
        print("JEPA State Decoder is disabled in the configuration.")

    # The EncoderDecoderJEPAStyle model is now initialized conditionally above,
    # and stored in models['std_enc_dec'] if selected.
    # The old models['enc_dec_jepa_style'] key is no longer used for this purpose.
    # If a separate instance for comparison is ever needed, it would be re-added here,
    # but for now, the logic handles selecting one or the other as the primary 'std_enc_dec'.

    # --- Initialize LARP Models ---
    larp_config_main = reward_pred_config.get('larp', {})
    enc_dec_larp_config = larp_config_main.get('encoder_decoder_larp', {})
    jepa_larp_config = larp_config_main.get('jepa_larp', {})

    larp_enc_dec = None
    if enc_dec_larp_config.get('enabled', False) and std_enc_dec:
        print("Initializing LARP for Encoder-Decoder...")
        # Determine if the std_enc_dec is 'standard' or 'jepa_style' for correct dim calculation
        # The 'encoder_decoder_variant' variable holds this information from earlier in the function
        current_encoder_decoder_variant = std_enc_dec_config.get('variant', 'standard') # Re-access or pass down

        larp_input_dim_enc_dec = calculate_larp_input_dim_enc_dec(
            config=config, # Full config
            encoder_decoder_variant=current_encoder_decoder_variant,
            image_h_w=image_h_w,
            input_channels=input_channels
        )
        larp_enc_dec = LookAheadRewardPredictorMLP(
            input_dim=larp_input_dim_enc_dec,
            hidden_dims=enc_dec_larp_config.get('hidden_dims', [512, 256, 128]),
            activation_fn_str=enc_dec_larp_config.get('activation', 'relu'),
            use_batch_norm=enc_dec_larp_config.get('use_batch_norm', False),
            dropout_rate=enc_dec_larp_config.get('dropout_rate', 0.0)
        ).to(device)
    models['larp_enc_dec'] = larp_enc_dec

    larp_jepa = None
    if jepa_larp_config.get('enabled', False) and jepa_model:
        print("Initializing LARP for JEPA...")
        larp_input_dim_jepa = calculate_larp_input_dim_jepa(config=config) # Full config
        larp_jepa = LookAheadRewardPredictorMLP(
            input_dim=larp_input_dim_jepa,
            hidden_dims=jepa_larp_config.get('hidden_dims', [512, 256, 128]),
            activation_fn_str=jepa_larp_config.get('activation', 'relu'),
            use_batch_norm=jepa_larp_config.get('use_batch_norm', False),
            dropout_rate=jepa_larp_config.get('dropout_rate', 0.0)
        ).to(device)
    models['larp_jepa'] = larp_jepa

    return models
