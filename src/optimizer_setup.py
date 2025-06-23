# Contents for src/optimizer_setup.py
import torch
import torch.optim as optim

def initialize_optimizers(models_map, config): # Renamed models to models_map for clarity
    optimizers = {}

    # Extract config sections
    training_config = config.get('training', {})
    models_config = config.get('models', {})
    
    # General learning rate from training config
    general_lr = training_config.get('learning_rate', 0.0003)

    # Optimizer for Standard Encoder-Decoder
    std_enc_dec_model = models_map.get('std_enc_dec')
    if std_enc_dec_model:
        optimizer_std_enc_dec = optim.AdamW(
            std_enc_dec_model.parameters(),
            lr=general_lr
        )
        optimizers['std_enc_dec'] = optimizer_std_enc_dec
        print(f"Standard Encoder-Decoder optimizer initialized with LR: {general_lr}")

    # Optimizer for JEPA
    jepa_model = models_map.get('jepa')
    if jepa_model:
        jepa_config = models_config.get('jepa', {})
        lr_jepa = jepa_config.get('learning_rate', general_lr)
        optimizer_jepa = optim.AdamW(
            jepa_model.parameters(),
            lr=lr_jepa
        )
        optimizers['jepa'] = optimizer_jepa
        print(f"JEPA optimizer initialized with LR: {lr_jepa}")

    # Optimizer for Encoder-Decoder Reward MLP (if enabled and model exists)
    reward_pred_config = models_config.get('reward_predictors', {})
    enc_dec_mlp_config = reward_pred_config.get('encoder_decoder_reward_mlp', {})
    reward_mlp_enc_dec_model = models_map.get('reward_mlp_enc_dec')
    if enc_dec_mlp_config.get('enabled', False) and reward_mlp_enc_dec_model:
        lr_enc_dec_reward = enc_dec_mlp_config.get('learning_rate', 0.0003)
        optimizer_reward_mlp_enc_dec = optim.AdamW(
            reward_mlp_enc_dec_model.parameters(),
            lr=lr_enc_dec_reward
        )
        optimizers['reward_mlp_enc_dec'] = optimizer_reward_mlp_enc_dec
        print(f"Encoder-Decoder Reward MLP optimizer initialized with LR: {lr_enc_dec_reward}")
    else:
        optimizers['reward_mlp_enc_dec'] = None

    # Optimizer for JEPA Reward MLP (if enabled and model exists)
    jepa_mlp_config = reward_pred_config.get('jepa_reward_mlp', {})
    reward_mlp_jepa_model = models_map.get('reward_mlp_jepa')
    if jepa_mlp_config.get('enabled', False) and reward_mlp_jepa_model:
        lr_jepa_reward = jepa_mlp_config.get('learning_rate', 0.0003)
        optimizer_reward_mlp_jepa = optim.AdamW(
            reward_mlp_jepa_model.parameters(),
            lr=lr_jepa_reward
        )
        optimizers['reward_mlp_jepa'] = optimizer_reward_mlp_jepa
        print(f"JEPA Reward MLP optimizer initialized with LR: {lr_jepa_reward}")
    else:
        optimizers['reward_mlp_jepa'] = None

    # Optimizer for JEPA State Decoder (if model exists)
    jepa_decoder_model = models_map.get('jepa_decoder')
    if jepa_decoder_model:
        jepa_config = models_config.get('jepa', {})
        jepa_decoder_training_config = jepa_config.get('decoder_training', {})
        # Fallback: jepa_decoder LR -> general LR -> default 0.0003
        learning_rate_jepa_decoder = jepa_decoder_training_config.get('learning_rate', general_lr)

        optimizer_jepa_decoder = torch.optim.AdamW(
            jepa_decoder_model.parameters(),
            lr=learning_rate_jepa_decoder
        )
        optimizers['jepa_decoder'] = optimizer_jepa_decoder
        print(f"JEPA State Decoder optimizer initialized with LR: {learning_rate_jepa_decoder}")
    else:
        optimizers['jepa_decoder'] = None
        # No message needed if decoder itself is None, model_setup would have printed it's disabled.

    return optimizers
