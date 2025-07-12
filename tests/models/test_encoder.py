import pytest
import torch
from src.models.encoder import (
    RotaryEmbedding,
    FactorizedPatchEmbed,
    MLP,
    MultiHeadSelfAttention,
    TransformerBlock,
    VideoViT,
)

# Test parameters
BATCH_SIZE = 2
FRAMES = 4
IMG_H, IMG_W = 64, 64
PATCH_H, PATCH_W = 8, 8
EMBED_DIM = 64
NUM_HEADS = 4
MLP_RATIO = 4.0
SEQ_LEN = (IMG_H // PATCH_H) * (IMG_W // PATCH_W)

@pytest.fixture
def dummy_video_batch():
    """Provides a dummy video tensor."""
    return torch.randn(BATCH_SIZE, 1, FRAMES, IMG_H, IMG_W)

def test_rotary_embedding_shape():
    """Tests the output shape of the RotaryEmbedding module."""
    rotary_emb = RotaryEmbedding(dim=EMBED_DIM // NUM_HEADS)
    cos, sin = rotary_emb(seq_len=SEQ_LEN, device='cpu')
    
    expected_shape = (1, 1, SEQ_LEN, EMBED_DIM // NUM_HEADS)
    assert cos.shape == expected_shape, f"Expected cos shape {expected_shape}, but got {cos.shape}"
    assert sin.shape == expected_shape, f"Expected sin shape {expected_shape}, but got {sin.shape}"

def test_factorized_patch_embed_shape(dummy_video_batch):
    """Tests the output shape of the FactorizedPatchEmbed module."""
    patch_embed = FactorizedPatchEmbed(patch_size=(PATCH_H, PATCH_W), embed_dim=EMBED_DIM)
    output = patch_embed(dummy_video_batch)
    
    expected_shape = (BATCH_SIZE, FRAMES, SEQ_LEN, EMBED_DIM)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_mlp_shape():
    """Tests that the MLP module preserves the input tensor's shape."""
    mlp = MLP(embed_dim=EMBED_DIM, mlp_ratio=MLP_RATIO)
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    output = mlp(input_tensor)
    
    assert input_tensor.shape == output.shape, f"Expected shape {input_tensor.shape}, but got {output.shape}"

def test_multi_head_self_attention_shape():
    """Tests that the MultiHeadSelfAttention module preserves the input tensor's shape."""
    # Head dimension must be even for RoPE
    head_dim = EMBED_DIM // NUM_HEADS
    if head_dim % 2 != 0:
        pytest.skip("head_dim must be even for this test")
        
    attention = MultiHeadSelfAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    output = attention(input_tensor)
    
    assert input_tensor.shape == output.shape, f"Expected shape {input_tensor.shape}, but got {output.shape}"

def test_transformer_block_shape():
    """Tests that the TransformerBlock preserves the input tensor's shape."""
    # Head dimension must be even for RoPE
    head_dim = EMBED_DIM // NUM_HEADS
    if head_dim % 2 != 0:
        pytest.skip("head_dim must be even for this test")

    transformer_block = TransformerBlock(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    output = transformer_block(input_tensor)
    
    assert input_tensor.shape == output.shape, f"Expected shape {input_tensor.shape}, but got {output.shape}"

def test_videovit_forward_pass_shapes(dummy_video_batch):
    """
    Tests the shapes at critical points in the VideoViT's forward pass.
    """
    # Head dimension must be even for RoPE
    head_dim = EMBED_DIM // NUM_HEADS
    if head_dim % 2 != 0:
        pytest.skip("head_dim must be even for this test")

    model = VideoViT(
        img_h=IMG_H, img_w=IMG_W,
        frames_per_clip=FRAMES,
        patch_size_h=PATCH_H, patch_size_w=PATCH_W,
        embed_dim=EMBED_DIM,
        encoder_num_heads=NUM_HEADS
    )

    intermediate_shapes = {}

    def get_hook(name):
        def hook(model, input, output):
            intermediate_shapes[name] = output.shape
        return hook

    # Register hooks
    model.patch_embed.register_forward_hook(get_hook('patch_embed'))
    model.blocks[0].register_forward_hook(get_hook('first_block'))
    model.norm.register_forward_hook(get_hook('norm'))

    # Forward pass
    final_output = model(dummy_video_batch)

    # 1. After patch embedding
    expected_patch_embed_shape = (BATCH_SIZE, FRAMES, SEQ_LEN, EMBED_DIM)
    assert intermediate_shapes['patch_embed'] == expected_patch_embed_shape, \
        f"Shape after patch_embed is incorrect. Expected {expected_patch_embed_shape}, got {intermediate_shapes['patch_embed']}"

    # 2. After first transformer block (and reshape)
    expected_block_input_shape = (BATCH_SIZE * FRAMES, SEQ_LEN, EMBED_DIM)
    assert intermediate_shapes['first_block'] == expected_block_input_shape, \
        f"Shape after first block is incorrect. Expected {expected_block_input_shape}, got {intermediate_shapes['first_block']}"

    # 3. After final LayerNorm
    expected_norm_shape = (BATCH_SIZE * FRAMES, SEQ_LEN, EMBED_DIM)
    assert intermediate_shapes['norm'] == expected_norm_shape, \
        f"Shape after norm is incorrect. Expected {expected_norm_shape}, got {intermediate_shapes['norm']}"

    # 4. Final output shape
    expected_final_shape = (BATCH_SIZE, FRAMES, EMBED_DIM)
    assert final_output.shape == expected_final_shape, \
        f"Final output shape is incorrect. Expected {expected_final_shape}, got {final_output.shape}"
