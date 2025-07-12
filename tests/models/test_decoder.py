import pytest
import torch
import torch.nn as nn
from src.models.decoder import HybridConvTransformerDecoder


class TestHybridConvTransformerDecoder:
    """Test suite for HybridConvTransformerDecoder with single token input."""
    
    @pytest.fixture
    def decoder_config(self):
        """Default decoder configuration for testing."""
        return {
            'img_h': 64,
            'img_w': 64,
            'frames_per_clip': 1,
            'embed_dim': 768,
            'decoder_embed_dim': 512,
            'decoder_num_heads': 8,
            'num_upsampling_blocks': 3,
            'patch_size_h': 8,
            'patch_size_w': 8
        }
    
    @pytest.fixture
    def decoder(self, decoder_config):
        """Create decoder instance for testing."""
        return HybridConvTransformerDecoder(**decoder_config)
    
    def test_decoder_initialization(self, decoder, decoder_config):
        """Test that decoder initializes correctly."""
        expected_h_p = decoder_config['img_h'] // decoder_config['patch_size_h']
        expected_w_p = decoder_config['img_w'] // decoder_config['patch_size_w']
        
        assert decoder.H_p == expected_h_p
        assert decoder.W_p == expected_w_p
        assert decoder.decoder_embed_dim == decoder_config['decoder_embed_dim']
        assert decoder.embed_dim == decoder_config['embed_dim']
        
        # Check token_decoder layer
        assert isinstance(decoder.token_decoder, nn.Linear)
        assert decoder.token_decoder.in_features == decoder_config['embed_dim']
        assert decoder.token_decoder.out_features == expected_h_p * expected_w_p * decoder_config['embed_dim']
        
        # Check projection layer
        assert isinstance(decoder.proj, nn.Linear)
        assert decoder.proj.in_features == decoder_config['embed_dim']
        assert decoder.proj.out_features == decoder_config['decoder_embed_dim']
    
    def test_forward_pass_shape(self, decoder, decoder_config):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        latent_token = torch.randn(batch_size, 1, decoder_config['embed_dim'])
        
        output = decoder(latent_token)
        
        expected_shape = (batch_size, 1, 1, decoder_config['img_h'], decoder_config['img_w'])
        assert output.shape == expected_shape
    
    def test_forward_pass_single_token_requirement(self, decoder, decoder_config):
        """Test that decoder enforces single token input."""
        batch_size = 2
        
        # Test with correct single token
        valid_input = torch.randn(batch_size, 1, decoder_config['embed_dim'])
        output = decoder(valid_input)
        assert output.shape == (batch_size, 1, 1, decoder_config['img_h'], decoder_config['img_w'])
        
        # Test with multiple tokens should raise assertion error
        invalid_input = torch.randn(batch_size, 5, decoder_config['embed_dim'])
        with pytest.raises(AssertionError, match="Expected single token"):
            decoder(invalid_input)
    
    def test_different_batch_sizes(self, decoder, decoder_config):
        """Test decoder with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            latent_token = torch.randn(batch_size, 1, decoder_config['embed_dim'])
            output = decoder(latent_token)
            
            expected_shape = (batch_size, 1, 1, decoder_config['img_h'], decoder_config['img_w'])
            assert output.shape == expected_shape
    
    def test_different_configurations(self):
        """Test decoder with different configuration parameters."""
        configs = [
            {'img_h': 32, 'img_w': 32, 'patch_size_h': 4, 'patch_size_w': 4, 'embed_dim': 256},
            {'img_h': 128, 'img_w': 128, 'patch_size_h': 16, 'patch_size_w': 16, 'embed_dim': 1024},
            {'img_h': 64, 'img_w': 64, 'patch_size_h': 8, 'patch_size_w': 8, 'embed_dim': 512}
        ]
        
        for config in configs:
            decoder = HybridConvTransformerDecoder(
                img_h=config['img_h'],
                img_w=config['img_w'],
                embed_dim=config['embed_dim'],
                patch_size_h=config['patch_size_h'],
                patch_size_w=config['patch_size_w']
            )
            
            batch_size = 2
            latent_token = torch.randn(batch_size, 1, config['embed_dim'])
            output = decoder(latent_token)
            
            expected_shape = (batch_size, 1, 1, config['img_h'], config['img_w'])
            assert output.shape == expected_shape
    
    def test_gradient_flow(self, decoder, decoder_config):
        """Test that gradients flow properly through the decoder."""
        batch_size = 2
        latent_token = torch.randn(batch_size, 1, decoder_config['embed_dim'], requires_grad=True)
        
        output = decoder(latent_token)
        loss = output.mean()
        loss.backward()
        
        # Check that input gradients exist
        assert latent_token.grad is not None
        assert not torch.allclose(latent_token.grad, torch.zeros_like(latent_token.grad))
        
        # Check that decoder parameters have gradients
        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_deterministic_output(self, decoder, decoder_config):
        """Test that decoder produces deterministic output for same input."""
        batch_size = 2
        latent_token = torch.randn(batch_size, 1, decoder_config['embed_dim'])
        
        # Set to eval mode to ensure deterministic behavior
        decoder.eval()
        
        with torch.no_grad():
            output1 = decoder(latent_token)
            output2 = decoder(latent_token)
            
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_spatial_token_generation(self, decoder, decoder_config):
        """Test internal spatial token generation logic."""
        batch_size = 3
        latent_token = torch.randn(batch_size, 1, decoder_config['embed_dim'])
        
        # Extract the token and test token_decoder
        token = latent_token.squeeze(1)  # [B, E]
        spatial_tokens_flat = decoder.token_decoder(token)
        
        expected_flat_size = decoder.H_p * decoder.W_p * decoder_config['embed_dim']
        assert spatial_tokens_flat.shape == (batch_size, expected_flat_size)
        
        # Test reshaping
        spatial_tokens = spatial_tokens_flat.reshape(batch_size, decoder.H_p * decoder.W_p, decoder_config['embed_dim'])
        expected_spatial_shape = (batch_size, decoder.H_p * decoder.W_p, decoder_config['embed_dim'])
        assert spatial_tokens.shape == expected_spatial_shape
    
    def test_output_range_and_type(self, decoder, decoder_config):
        """Test that output is properly formed tensor."""
        batch_size = 2
        latent_token = torch.randn(batch_size, 1, decoder_config['embed_dim'])
        
        output = decoder(latent_token)
        
        # Check type
        assert isinstance(output, torch.Tensor)
        assert output.dtype == torch.float32
        
        # Check that output contains finite values
        assert torch.isfinite(output).all()
        
        # Check that output is not all zeros (model should produce some meaningful output)
        assert not torch.allclose(output, torch.zeros_like(output))


if __name__ == "__main__":
    # Run a quick test
    config = {
        'img_h': 64,
        'img_w': 64,
        'embed_dim': 768,
        'decoder_embed_dim': 512
    }
    
    decoder = HybridConvTransformerDecoder(**config)
    latent_token = torch.randn(2, 1, 768)
    
    try:
        output = decoder(latent_token)
        print(f"Test passed! Output shape: {output.shape}")
        print("Expected shape: (2, 1, 1, 64, 64)")
        print(f"Model parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    except Exception as e:
        print(f"Test failed: {e}")
