import torch
import pytest
from src.models.predictor import MLPHistoryPredictor
from src.utils.init_models import (
    init_mlp_predictor,
    init_mlp_predictor_from_config,
    init_mlp_predictor_from_encoder_output
)


class TestMLPHistoryPredictor:
    
    def test_init_default_config(self):
        """Test MLPHistoryPredictor initialization with default configuration."""
        model = MLPHistoryPredictor()
        
        assert model.frames_per_clip == 6
        assert model.latent_dim == 128
        assert model.num_actions == 18
        assert isinstance(model.activation, torch.nn.SiLU)
    
    def test_init_custom_config(self):
        """Test MLPHistoryPredictor initialization with custom configuration."""
        model = MLPHistoryPredictor(
            frames_per_clip=8,
            latent_dim=256,
            num_actions=10,
            hidden_sizes=[1024, 512, 256],
            activation='relu',
            dropout_rate=0.2
        )
        
        assert model.frames_per_clip == 8
        assert model.latent_dim == 256
        assert model.num_actions == 10
        assert isinstance(model.activation, torch.nn.ReLU)
    
    def test_init_via_init_models(self):
        """Test MLPHistoryPredictor initialization via init_models utility."""
        model = init_mlp_predictor()
        
        assert isinstance(model, MLPHistoryPredictor)
        assert model.frames_per_clip == 6
        assert model.latent_dim == 128
    
    def test_init_from_config(self):
        """Test MLPHistoryPredictor initialization from config file."""
        model = init_mlp_predictor_from_config()
        
        assert isinstance(model, MLPHistoryPredictor)
        # Should load num_actions from config.yaml (which is 6)
        assert model.num_actions == 6
    
    def test_from_encoder_output(self):
        """Test MLPHistoryPredictor.from_encoder_output class method."""
        # Simulate encoder output
        encoder_output = torch.randn(4, 8, 256)  # [B=4, T=8, E=256]
        
        model = MLPHistoryPredictor.from_encoder_output(encoder_output)
        
        assert model.frames_per_clip == 8  # Inferred from T
        assert model.latent_dim == 256      # Inferred from E
        assert model.num_actions == 6       # From config (default config.yaml)
    
    def test_init_from_encoder_output_via_init_models(self):
        """Test initialization from encoder output via init_models utility."""
        encoder_output = torch.randn(2, 10, 512)  # [B=2, T=10, E=512]
        
        model = init_mlp_predictor_from_encoder_output(encoder_output)
        
        assert model.frames_per_clip == 10  # Inferred from T
        assert model.latent_dim == 512      # Inferred from E
        assert model.num_actions == 6       # From config
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        model = MLPHistoryPredictor()
        
        B, T, E = 4, 6, 128
        x = torch.randn(B, T, E)
        a = torch.randint(0, 18, (B, T))
        
        output = model(x, a)
        
        assert output.shape == (B, 1, E)
    
    def test_forward_with_inferred_model(self):
        """Test forward pass with model created from encoder output."""
        # Create model from encoder output
        encoder_output = torch.randn(3, 5, 200)
        model = MLPHistoryPredictor.from_encoder_output(encoder_output)
        
        # Test forward pass
        x = torch.randn(3, 5, 200)
        a = torch.randint(0, 6, (3, 5))
        
        output = model(x, a)
        assert output.shape == (3, 1, 200)
    
    def test_forward_dummy_input(self):
        """Test forward pass with dummy input as specified in requirements."""
        model = MLPHistoryPredictor()
        
        # Dummy input as specified
        x = torch.randn(4, 6, 128)       # [B=4, T=6, E=128]
        a = torch.randint(0, 5, (4, 6))  # 5 discrete actions
        
        output = model(x, a)
        
        print(f"Output shape: {output.shape}")  # Should be [4, 1, 128]
        assert output.shape == (4, 1, 128)
    
    def test_forward_validation(self):
        """Test that forward pass validates input dimensions."""
        model = MLPHistoryPredictor(frames_per_clip=6, latent_dim=128)
        
        # Wrong number of frames
        x_wrong_frames = torch.randn(2, 8, 128)  # Should be 6 frames
        a = torch.randint(0, 18, (2, 8))
        
        with pytest.raises(ValueError, match="Expected 6 frames, got 8"):
            model(x_wrong_frames, a)
        
        # Wrong latent dimension
        x_wrong_latent = torch.randn(2, 6, 256)  # Should be 128 latent dim
        a = torch.randint(0, 18, (2, 6))
        
        with pytest.raises(ValueError, match="Expected latent dim 128, got 256"):
            model(x_wrong_latent, a)
    
    def test_invalid_encoder_output_shape(self):
        """Test that invalid encoder output shape raises error."""
        # Wrong number of dimensions
        invalid_output = torch.randn(4, 6)  # Should be 3D
        
        with pytest.raises(ValueError, match="Expected encoder output shape"):
            MLPHistoryPredictor.from_encoder_output(invalid_output)
    
    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = MLPHistoryPredictor()
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 6, 128)
            a = torch.randint(0, 18, (batch_size, 6))
            
            output = model(x, a)
            assert output.shape == (batch_size, 1, 128)
    
    def test_forward_different_latent_dims(self):
        """Test forward pass with different latent dimensions."""
        for latent_dim in [64, 256, 512]:
            model = MLPHistoryPredictor(latent_dim=latent_dim)
            
            x = torch.randn(2, 6, latent_dim)
            a = torch.randint(0, 18, (2, 6))
            
            output = model(x, a)
            assert output.shape == (2, 1, latent_dim)
    
    def test_device_compatibility(self):
        """Test model can be moved to different devices."""
        model = MLPHistoryPredictor()
        
        # Test CPU
        x = torch.randn(2, 6, 128)
        a = torch.randint(0, 18, (2, 6))
        output = model(x, a)
        assert output.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            a = a.cuda()
            output = model(x, a)
            assert output.device.type == 'cuda'
    
    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ['silu', 'relu', 'gelu']
        expected_types = [torch.nn.SiLU, torch.nn.ReLU, torch.nn.GELU]
        
        for activation, expected_type in zip(activations, expected_types):
            model = MLPHistoryPredictor(activation=activation)
            assert isinstance(model.activation, expected_type)
    
    def test_invalid_activation(self):
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            MLPHistoryPredictor(activation='invalid')
    
    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = MLPHistoryPredictor()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Should have reasonable number of parameters
        assert total_params > 100000  # At least 100k parameters
        assert total_params < 10000000  # Less than 10M parameters
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = MLPHistoryPredictor()
        
        x = torch.randn(2, 6, 128, requires_grad=True)
        a = torch.randint(0, 18, (2, 6))
        
        output = model(x, a)
        loss = output.sum()
        loss.backward()
        
        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    # Run tests with different initialization methods
    print("Testing MLPHistoryPredictor with various initialization methods...")
    
    # Test 1: Default initialization
    print("\n1. Default initialization:")
    model = MLPHistoryPredictor()
    x = torch.randn(4, 6, 128)
    a = torch.randint(0, 5, (4, 6))
    output = model(x, a)
    print(f"   Output shape: {output.shape}")
    
    # Test 2: From config
    print("\n2. From config:")
    model_config = init_mlp_predictor_from_config()
    print(f"   num_actions from config: {model_config.num_actions}")
    output_config = model_config(x, a)
    print(f"   Output shape: {output_config.shape}")
    
    # Test 3: From encoder output
    print("\n3. From encoder output:")
    encoder_output = torch.randn(4, 8, 256)  # Different dimensions
    model_inferred = MLPHistoryPredictor.from_encoder_output(encoder_output)
    print(f"   Inferred frames_per_clip: {model_inferred.frames_per_clip}")
    print(f"   Inferred latent_dim: {model_inferred.latent_dim}")
    print(f"   num_actions from config: {model_inferred.num_actions}")
    
    x_inferred = torch.randn(4, 8, 256)
    a_inferred = torch.randint(0, 6, (4, 8))
    output_inferred = model_inferred(x_inferred, a_inferred)
    print(f"   Output shape: {output_inferred.shape}")
    
    print("\n✓ All tests passed!") 