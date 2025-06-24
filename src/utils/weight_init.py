import torch.nn as nn

def initialize_weights(module):
    """
    Initializes weights for layers in a PyTorch module.

    Args:
        module (nn.Module): The module whose layers will be initialized.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming Normal initialization for Conv2d layers
            # Recommended for layers followed by ReLU or LeakyReLU activations
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Kaiming Normal initialization for Linear layers
            # Recommended for layers followed by ReLU or GELU activations
            # For other activations like Tanh or Sigmoid, Xavier/Glorot might be preferred
            # but most modern networks use ReLU/GELU.
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # Example for Xavier/Glorot if needed for specific linear layers:
            # if hasattr(m, 'activation') and m.activation == 'tanh':
            #     nn.init.xavier_normal_(m.weight)
            # else:
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # if m.bias is not None:
            #     nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Embedding):
            # Initialize embedding layers with a normal distribution (small std dev)
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm weights to 1 and biases to 0
            # This is a common practice for LayerNorm initialization
            if m.elementwise_affine: # LayerNorm has learnable affine parameters
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        # Note: For nn.Parameter tensors that are not part of a standard layer (e.g., ViT's cls_token, pos_embedding),
        # they need to be initialized manually in their respective module's __init__ method,
        # as module.apply() or module.modules() won't directly process them in the same way as nn.Module instances.

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_num_parameters(module, check_total=True):
    n_encoder = 0
    n_online_encoder = 0
    n_action_embedding = 0
    n_predictor = 0
    n_decoder = 0
    if hasattr(module, 'encoder'):
        n_encoder = count_parameters(module.encoder)
        print(f'{"Num parameters in encoder:":<65}{n_encoder:,}')
    if hasattr(module, 'online_encoder'):
        n_online_encoder = count_parameters(module.online_encoder)
        print(f'{"Num parameters in online encoder:":<65}{n_online_encoder:,}')
    if hasattr(module, 'action_embedding'):
        n_action_embedding = count_parameters(module.action_embedding)
        print(f'{"Num parameters in action embedding:":<65}{n_action_embedding:,}')
    if hasattr(module, 'predictor'):
        n_predictor = count_parameters(module.predictor)
        print(f'{"Num parameters in predictor:":<65}{n_predictor:,}')
    if hasattr(module, 'decoder'):
        n_decoder = count_parameters(module.decoder)
        print(f'{"Num parameters in decoder:":<65}{n_decoder:,}')
    total_params = n_encoder + n_online_encoder + n_action_embedding + n_predictor + n_decoder
    total_count = count_parameters(module)
    if check_total:
        assert total_params == total_count, f"Total parameters mismatch: {total_params} != {total_count}"
    print(f'{"Total num parameters:":<65}{total_count:,}')
