# Contents for src/optimizer_setup.py
import torch.optim as optim

def initialize_optimizers(models, config):
    optimizers = {}

    # Optimizer for Standard Encoder-Decoder
    if models.get('std_enc_dec'):
        optimizer_std_enc_dec = optim.AdamW(
            models['std_enc_dec'].parameters(),
            lr=config.get('learning_rate', 0.0003)
        )
        optimizers['std_enc_dec'] = optimizer_std_enc_dec

    # Optimizer for JEPA
    if models.get('jepa'):
        optimizer_jepa = optim.AdamW(
            models['jepa'].parameters(),
            lr=config.get('learning_rate_jepa', config.get('learning_rate', 0.0003))
        )
        optimizers['jepa'] = optimizer_jepa

    # Optimizer for Encoder-Decoder Reward MLP (if enabled and model exists)
    reward_pred_config = config.get('reward_predictors', {})
    enc_dec_mlp_config = reward_pred_config.get('encoder_decoder_reward_mlp', {})
    if enc_dec_mlp_config.get('enabled', False) and models.get('reward_mlp_enc_dec'):
        optimizer_reward_mlp_enc_dec = optim.AdamW(
            models['reward_mlp_enc_dec'].parameters(),
            lr=enc_dec_mlp_config.get('learning_rate', 0.0003)
        )
        optimizers['reward_mlp_enc_dec'] = optimizer_reward_mlp_enc_dec
    else:
        optimizers['reward_mlp_enc_dec'] = None


    # Optimizer for JEPA Reward MLP (if enabled and model exists)
    jepa_mlp_config = reward_pred_config.get('jepa_reward_mlp', {})
    if jepa_mlp_config.get('enabled', False) and models.get('reward_mlp_jepa'):
        optimizer_reward_mlp_jepa = optim.AdamW(
            models['reward_mlp_jepa'].parameters(),
            lr=jepa_mlp_config.get('learning_rate', 0.0003)
        )
        optimizers['reward_mlp_jepa'] = optimizer_reward_mlp_jepa
    else:
        optimizers['reward_mlp_jepa'] = None

    return optimizers
