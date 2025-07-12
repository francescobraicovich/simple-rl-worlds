import torch
import pytest
from src.models.reward_predictor import RewardPredictor

@pytest.fixture
def reward_predictor():
    embedding_dim = 128
    internal_embedding_dim = 64
    num_heads = 4
    return RewardPredictor(embedding_dim, internal_embedding_dim, num_heads)

def test_reward_predictor_output_shape(reward_predictor):
    batch_size = 2
    embedding_dim = reward_predictor.embedding_dim

    x1 = torch.randn(batch_size, 1, embedding_dim)
    x2 = torch.randn(batch_size, 2, embedding_dim)

    reward = reward_predictor(x1, x2)

    assert reward.shape == (batch_size, 1, 1)

def test_reward_predictor_learnable_cls_token(reward_predictor):
    initial_cls_token = reward_predictor.cls_token.clone()
    # Perform a dummy forward pass to ensure parameters are initialized
    x1 = torch.randn(1, 1, reward_predictor.embedding_dim)
    x2 = torch.randn(1, 2, reward_predictor.embedding_dim)
    _ = reward_predictor(x1, x2)
    # Check if cls_token is a learnable parameter
    assert reward_predictor.cls_token.requires_grad

def test_reward_predictor_projection_layers(reward_predictor):
    embedding_dim = reward_predictor.embedding_dim
    internal_embedding_dim = reward_predictor.internal_embedding_dim

    # Check if projection layers exist and have correct input/output dimensions
    assert isinstance(reward_predictor.proj_x1, torch.nn.Linear)
    assert reward_predictor.proj_x1.in_features == embedding_dim
    assert reward_predictor.proj_x1.out_features == internal_embedding_dim

    assert isinstance(reward_predictor.proj_x2, torch.nn.Linear)
    assert reward_predictor.proj_x2.in_features == embedding_dim
    assert reward_predictor.proj_x2.out_features == internal_embedding_dim

def test_reward_predictor_mlp_head(reward_predictor):
    internal_embedding_dim = reward_predictor.internal_embedding_dim

    # Check if MLP head exists and has correct input/output dimensions
    assert isinstance(reward_predictor.mlp_head, torch.nn.Sequential)
    assert isinstance(reward_predictor.mlp_head[0], torch.nn.Linear)
    assert reward_predictor.mlp_head[0].in_features == internal_embedding_dim
    assert reward_predictor.mlp_head[0].out_features == internal_embedding_dim // 2
    assert isinstance(reward_predictor.mlp_head[2], torch.nn.Linear)
    assert reward_predictor.mlp_head[2].in_features == internal_embedding_dim // 2
    assert reward_predictor.mlp_head[2].out_features == 1
