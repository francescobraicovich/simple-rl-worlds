import pytest
import torch
from src.models.predictor import LatentDynamicsPredictor

# Test parameters
BATCH_SIZE = 2
FRAMES_PER_CLIP = 4
EMBED_DIM = 64
NUM_ACTIONS = 18
NUM_HEADS = 4

@pytest.fixture
def dummy_latent_vectors():
    """Provides a dummy latent vector tensor."""
    return torch.randn(BATCH_SIZE, FRAMES_PER_CLIP, EMBED_DIM)

@pytest.fixture
def dummy_actions():
    """Provides a dummy action tensor."""
    return torch.randint(0, NUM_ACTIONS, (BATCH_SIZE,))

def test_latent_dynamics_predictor_shapes(dummy_latent_vectors, dummy_actions):
    """
    Tests the shapes at critical points in the LatentDynamicsPredictor's forward pass.
    """
    # Head dimension must be even for RoPE in the underlying TransformerBlocks
    head_dim = EMBED_DIM // NUM_HEADS
    if head_dim % 2 != 0:
        pytest.skip("head_dim must be even for this test")

    model = LatentDynamicsPredictor(
        frames_per_clip=FRAMES_PER_CLIP,
        embed_dim=EMBED_DIM,
        num_actions=NUM_ACTIONS,
        predictor_num_heads=NUM_HEADS
    )

    intermediate_tensors = {}

    def get_hook(name):
        def hook(model, input, output):
            intermediate_tensors[name] = output
        return hook

    def get_pre_hook(name):
        def hook(model, input):
            intermediate_tensors[name] = input
        return hook

    # Register hooks to capture intermediate tensor shapes
    # We'll hook into the first block's input to check the shape after action embedding and concatenation
    model.blocks[0].register_forward_pre_hook(get_pre_hook('transformer_input'))
    model.blocks[-1].register_forward_hook(get_hook('transformer_output'))
    model.prediction_head.register_forward_hook(get_hook('prediction_head_output'))

    # Forward pass
    x_pred = model(dummy_latent_vectors, dummy_actions)

    # 1. Test transformer input shape (after action embedding and concatenation)
    expected_transformer_input_shape = (BATCH_SIZE, FRAMES_PER_CLIP + 1, EMBED_DIM)
    print(f"Expected transformer input shape: {expected_transformer_input_shape}")
    # The pre-hook for blocks[0] input gives a tuple (input_tensor,)
    assert intermediate_tensors['transformer_input'][0].shape == expected_transformer_input_shape, \
        f"Transformer input shape is incorrect. Expected {expected_transformer_input_shape}, got {intermediate_tensors['transformer_input'][0].shape}"

    # 2. Test transformer output shape (output of the last block)
    expected_transformer_output_shape = (BATCH_SIZE, FRAMES_PER_CLIP + 1, EMBED_DIM)
    assert intermediate_tensors['transformer_output'].shape == expected_transformer_output_shape, \
        f"Transformer output shape is incorrect. Expected {expected_transformer_output_shape}, got {intermediate_tensors['transformer_output'].shape}"

    # 3. Test prediction head output shape
    expected_head_output_shape = (BATCH_SIZE, 1, EMBED_DIM)
    assert intermediate_tensors['prediction_head_output'].shape == expected_head_output_shape, f"Prediction head output shape is incorrect. Expected {expected_head_output_shape}, got {intermediate_tensors['prediction_head_output'].shape}"

    # 4. Final output shape
    assert x_pred.shape == expected_head_output_shape, \
        f"Final output shape is incorrect. Expected {expected_head_output_shape}, got {x_pred.shape}"