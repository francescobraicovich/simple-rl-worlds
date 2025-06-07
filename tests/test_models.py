import unittest
import torch

# Models to test
from models.cnn import CNNEncoder
from models.mlp import MLPEncoder
from models.vit import ViT
from models.encoder_decoder import StandardEncoderDecoder
from models.jepa import JEPA

class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_channels = 3
        self.image_size_int = 64
        self.image_size_tuple = (64, 48)
        self.latent_dim = 128
        self.dummy_img_square = torch.randn(self.batch_size, self.input_channels, self.image_size_int, self.image_size_int)
        self.dummy_img_rect = torch.randn(self.batch_size, self.input_channels, self.image_size_tuple[0], self.image_size_tuple[1])

    def test_cnn_encoder_initialization_and_forward(self):
        # Test with square image
        cnn_encoder = CNNEncoder(input_channels=self.input_channels, image_size=self.image_size_int, latent_dim=self.latent_dim)
        output = cnn_encoder(self.dummy_img_square)
        self.assertEqual(output.shape, (self.batch_size, self.latent_dim))

        # Test with rectangular image
        cnn_encoder_rect = CNNEncoder(input_channels=self.input_channels, image_size=self.image_size_tuple, latent_dim=self.latent_dim)
        output_rect = cnn_encoder_rect(self.dummy_img_rect)
        self.assertEqual(output_rect.shape, (self.batch_size, self.latent_dim))

        # Test with more parameters
        cnn_encoder_custom = CNNEncoder(
            input_channels=self.input_channels,
            image_size=self.image_size_int,
            latent_dim=self.latent_dim,
            num_conv_layers=2,
            base_filters=16,
            kernel_size=5,
            activation_fn_str='gelu',
            fc_hidden_dim=256
        )
        output_custom = cnn_encoder_custom(self.dummy_img_square)
        self.assertEqual(output_custom.shape, (self.batch_size, self.latent_dim))

    def test_mlp_encoder_initialization_and_forward(self):
        # Test with square image
        mlp_encoder = MLPEncoder(input_channels=self.input_channels, image_size=self.image_size_int, latent_dim=self.latent_dim)
        output = mlp_encoder(self.dummy_img_square)
        self.assertEqual(output.shape, (self.batch_size, self.latent_dim))

        # Test with rectangular image
        mlp_encoder_rect = MLPEncoder(input_channels=self.input_channels, image_size=self.image_size_tuple, latent_dim=self.latent_dim)
        output_rect = mlp_encoder_rect(self.dummy_img_rect)
        self.assertEqual(output_rect.shape, (self.batch_size, self.latent_dim))

        # Test with more parameters
        mlp_encoder_custom = MLPEncoder(
            input_channels=self.input_channels,
            image_size=self.image_size_int,
            latent_dim=self.latent_dim,
            num_hidden_layers=3,
            hidden_dim=1024,
            activation_fn_str='gelu'
        )
        output_custom = mlp_encoder_custom(self.dummy_img_square)
        self.assertEqual(output_custom.shape, (self.batch_size, self.latent_dim))


class TestModelSelection(unittest.TestCase):
    def setUp(self):
        self.image_size = 64
        self.patch_size = 16 # For ViT and decoder
        self.input_channels = 3
        self.action_dim = 5
        self.action_emb_dim = 32
        self.latent_dim = 128
        self.decoder_dim = 128
        self.decoder_depth = 2
        self.decoder_heads = 4
        self.decoder_mlp_dim = 256
        self.output_channels = 3
        self.output_image_size = 64

        # These params should be minimal and sufficient for the models' __init__
        # The models themselves have internal defaults for params not provided here.
        self.vit_encoder_params = {
            'patch_size': self.patch_size,
            'depth': 2,
            'heads': 4,
            'mlp_dim': 128,
            # 'pool', 'dropout', 'emb_dropout' will use ViT's internal defaults
        }
        self.cnn_encoder_params = {
            'num_conv_layers': 2,
            'base_filters': 16
            # Other CNN params will use CNNEncoder's internal defaults
        }
        self.mlp_encoder_params = {
            'num_hidden_layers': 1,
            'hidden_dim': 128
            # Other MLP params will use MLPEncoder's internal defaults
        }

        self.dummy_img = torch.randn(2, self.input_channels, self.image_size, self.image_size)
        self.dummy_action = torch.randn(2, self.action_dim) # Assuming continuous or already processed action

    def test_encoder_decoder_model_selection(self):
        # Test with ViT
        model_vit = StandardEncoderDecoder(
            image_size=self.image_size, patch_size=self.patch_size, input_channels=self.input_channels,
            action_dim=self.action_dim, action_emb_dim=self.action_emb_dim, latent_dim=self.latent_dim,
            decoder_dim=self.decoder_dim, decoder_depth=self.decoder_depth, decoder_heads=self.decoder_heads,
            decoder_mlp_dim=self.decoder_mlp_dim, output_channels=self.output_channels, output_image_size=self.output_image_size,
            encoder_type='vit', encoder_params=self.vit_encoder_params
        )
        self.assertIsInstance(model_vit.encoder, ViT)
        model_vit(self.dummy_img, self.dummy_action) # Check forward pass

        # Test with CNN
        model_cnn = StandardEncoderDecoder(
            image_size=self.image_size, patch_size=self.patch_size, input_channels=self.input_channels,
            action_dim=self.action_dim, action_emb_dim=self.action_emb_dim, latent_dim=self.latent_dim,
            decoder_dim=self.decoder_dim, decoder_depth=self.decoder_depth, decoder_heads=self.decoder_heads,
            decoder_mlp_dim=self.decoder_mlp_dim, output_channels=self.output_channels, output_image_size=self.output_image_size,
            encoder_type='cnn', encoder_params=self.cnn_encoder_params
        )
        self.assertIsInstance(model_cnn.encoder, CNNEncoder)
        model_cnn(self.dummy_img, self.dummy_action)

        # Test with MLP
        model_mlp = StandardEncoderDecoder(
            image_size=self.image_size, patch_size=self.patch_size, input_channels=self.input_channels,
            action_dim=self.action_dim, action_emb_dim=self.action_emb_dim, latent_dim=self.latent_dim,
            decoder_dim=self.decoder_dim, decoder_depth=self.decoder_depth, decoder_heads=self.decoder_heads,
            decoder_mlp_dim=self.decoder_mlp_dim, output_channels=self.output_channels, output_image_size=self.output_image_size,
            encoder_type='mlp', encoder_params=self.mlp_encoder_params
        )
        self.assertIsInstance(model_mlp.encoder, MLPEncoder)
        model_mlp(self.dummy_img, self.dummy_action)

    def test_jepa_model_selection(self):
        predictor_hidden_dim = 256
        # Test with ViT
        jepa_vit = JEPA(
            image_size=self.image_size, patch_size=self.patch_size, input_channels=self.input_channels,
            action_dim=self.action_dim, action_emb_dim=self.action_emb_dim, latent_dim=self.latent_dim,
            predictor_hidden_dim=predictor_hidden_dim, predictor_output_dim=self.latent_dim,
            encoder_type='vit', encoder_params=self.vit_encoder_params
        )
        self.assertIsInstance(jepa_vit.online_encoder, ViT)
        self.assertIsInstance(jepa_vit.target_encoder, ViT)
        jepa_vit(self.dummy_img, self.dummy_action, self.dummy_img) # Check forward pass

        # Test with CNN
        jepa_cnn = JEPA(
            image_size=self.image_size, patch_size=self.patch_size, input_channels=self.input_channels,
            action_dim=self.action_dim, action_emb_dim=self.action_emb_dim, latent_dim=self.latent_dim,
            predictor_hidden_dim=predictor_hidden_dim, predictor_output_dim=self.latent_dim,
            encoder_type='cnn', encoder_params=self.cnn_encoder_params
        )
        self.assertIsInstance(jepa_cnn.online_encoder, CNNEncoder)
        self.assertIsInstance(jepa_cnn.target_encoder, CNNEncoder)
        jepa_cnn(self.dummy_img, self.dummy_action, self.dummy_img)

        # Test with MLP
        jepa_mlp = JEPA(
            image_size=self.image_size, patch_size=self.patch_size, input_channels=self.input_channels,
            action_dim=self.action_dim, action_emb_dim=self.action_emb_dim, latent_dim=self.latent_dim,
            predictor_hidden_dim=predictor_hidden_dim, predictor_output_dim=self.latent_dim,
            encoder_type='mlp', encoder_params=self.mlp_encoder_params
        )
        self.assertIsInstance(jepa_mlp.online_encoder, MLPEncoder)
        self.assertIsInstance(jepa_mlp.target_encoder, MLPEncoder)
        jepa_mlp(self.dummy_img, self.dummy_action, self.dummy_img)

if __name__ == '__main__':
    unittest.main()
