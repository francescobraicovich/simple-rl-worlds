def calculate_larp_input_dim_enc_dec(config, encoder_decoder_variant, image_h_w, input_channels):
    """
    Calculates the input dimension for the Look-Ahead Reward Predictor (LARP)
    when used with an Encoder-Decoder base model (Standard or JEPA-Style).

    Args:
        config (dict): The main configuration dictionary.
        encoder_decoder_variant (str): Specifies the type of encoder-decoder ('standard' or 'jepa_style').
        image_h_w (int or tuple): Height and width of the input image. Can be a single int for square images.
        input_channels (int): Number of channels in the input image.

    Returns:
        int: The calculated input dimension for the LARP model.
    """
    models_config = config.get('models', {})
    shared_latent_dim = models_config.get('shared_latent_dim', 128)

    # Action embedding dimension from standard_encoder_decoder, assuming it's representative
    # or could be made a more general config if needed.
    std_enc_dec_config = models_config.get('standard_encoder_decoder', {})
    action_emb_dim = std_enc_dec_config.get('action_emb_dim', shared_latent_dim) # Default to shared_latent_dim if not specified

    # Predicted image dimensions
    # Assuming the predicted next state s_{t+1} by the world model has the same dimensions as the input state s_t.
    # The decoder in StandardEncoderDecoder outputs images of shape (output_channels, output_height, output_width).
    # For simplicity, we'll use input_channels and image_h_w as proxies for output shape.
    # A more robust solution might involve getting these from decoder_config if available.
    output_height, output_width = image_h_w
    # The problem description implies predicted s_{t+1} is an image.
    # Let's assume its channels are the same as input_channels for now.
    # If the world model's decoder outputs a different number of channels, this needs adjustment.
    #predicted_image_flat_dim = input_channels * output_height * output_width

    # Base dimension: encoded_s_t + predicted_s_t_plus_1_flat + action_embedding
    base_dim = shared_latent_dim + action_emb_dim


    if encoder_decoder_variant == 'jepa_style':
        # For JEPAStyle, add intermediate features from the predictor module.
        # The predictor module is part of the JEPAStateDecoder.
        # Its output dimension is typically the input_latent_dim to the decoder part of JEPAStateDecoder.
        jepa_state_decoder_arch_config = models_config.get('jepa_state_decoder_arch', {})
        # The predictor in EncoderDecoderJEPAStyle feeds into the decoder.
        # The 'predictor_output_dim' in the problem description refers to the output of this predictor.
        # Let's assume this predictor's output dimension is 'input_latent_dim' of the jepa_state_decoder_arch
        # or defaults to shared_latent_dim if not specified.
        predictor_output_dim = jepa_state_decoder_arch_config.get('input_latent_dim', shared_latent_dim)
        return base_dim + predictor_output_dim
    
    else:
        raise Warning("If encoder_decoder_variant is not 'jepa_style', the look-ahead reward predictor\n\
                       (LARP) will not be able to use the intermediate features from the predictor module.\n\
                       This makes LARP equivalent to a standard reward predictor.\n\
                       If this is not intended, please set encoder_decoder_variant to 'jepa_style'.")

    return base_dim

def calculate_larp_input_dim_jepa(config):
    """
    Calculates the input dimension for the Look-Ahead Reward Predictor (LARP)
    when used with a JEPA base model.

    Args:
        config (dict): The main configuration dictionary.

    Returns:
        int: The calculated input dimension for the LARP model.
    """
    models_config = config.get('models', {})
    shared_latent_dim = models_config.get('shared_latent_dim')

    std_enc_dec_config = models_config.get('standard_encoder_decoder', {})
    # Action embedding dimension from std_enc_dec_config
    action_emb_dim = std_enc_dec_config.get('action_emb_dim', shared_latent_dim) # Default to shared_latent_dim if not specified

    # For JEPA: encoded_s_t + predicted_latent_s_t_plus_1 + action_embedding
    # - encoded_s_t: shared_latent_dim (from online encoder)
    # - predicted_latent_s_t_plus_1: shared_latent_dim (output of JEPA's predictor)
    # - action_embedding: action_emb_dim
    return shared_latent_dim + shared_latent_dim + action_emb_dim
