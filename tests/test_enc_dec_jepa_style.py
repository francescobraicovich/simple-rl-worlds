import unittest
import torch
from src.models.encoder_decoder_jepa_style import EncoderDecoderJEPAStyle
from src.models.encoder_decoder import StandardEncoderDecoder
from src.models.jepa import JEPA
from src.models.mlp import RewardPredictorMLP

class TestEncoderDecoderJEPAStyle(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.img_size = 16
        self.input_channels = 3
        self.action_dim = 4
        self.action_emb_dim = 8
        self.latent_dim = 16
        self.predictor_hidden_dim = 12
        self.predictor_output_dim = 10
        self.decoder_dim = 10
        self.decoder_depth = 2
        self.decoder_heads = 2
        self.decoder_mlp_dim = 20
        self.decoder_patch_size = 4
        self.output_channels = 3
        self.device = torch.device('cpu')
        self.dummy_img = torch.randn(self.batch_size, self.input_channels, self.img_size, self.img_size)
        self.dummy_action = torch.randn(self.batch_size, self.action_dim)

    def test_forward_shape(self):
        model = EncoderDecoderJEPAStyle(
            image_size=self.img_size,
            patch_size=self.decoder_patch_size,
            input_channels=self.input_channels,
            action_dim=self.action_dim,
            action_emb_dim=self.action_emb_dim,
            latent_dim=self.latent_dim,
            predictor_hidden_dim=self.predictor_hidden_dim,
            predictor_output_dim=self.decoder_dim,
            predictor_dropout_rate=0.0,
            decoder_dim=self.decoder_dim,
            decoder_depth=self.decoder_depth,
            decoder_heads=self.decoder_heads,
            decoder_mlp_dim=self.decoder_mlp_dim,
            output_channels=self.output_channels,
            output_image_size=(self.img_size, self.img_size),
            decoder_dropout=0.0,
            encoder_type='mlp',
            encoder_params={'num_hidden_layers': 1, 'hidden_dim': 20, 'activation_fn_str': 'relu', 'dropout_rate': 0.0},
            decoder_patch_size=self.decoder_patch_size
        ).to(self.device)
        out = model(self.dummy_img, self.dummy_action)
        self.assertEqual(out.shape, (self.batch_size, self.output_channels, self.img_size, self.img_size))

    def test_loss_decreases(self):
        model = EncoderDecoderJEPAStyle(
            image_size=self.img_size,
            patch_size=self.decoder_patch_size,
            input_channels=self.input_channels,
            action_dim=self.action_dim,
            action_emb_dim=self.action_emb_dim,
            latent_dim=self.latent_dim,
            predictor_hidden_dim=self.predictor_hidden_dim,
            predictor_output_dim=self.decoder_dim,
            predictor_dropout_rate=0.0,
            decoder_dim=self.decoder_dim,
            decoder_depth=self.decoder_depth,
            decoder_heads=self.decoder_heads,
            decoder_mlp_dim=self.decoder_mlp_dim,
            output_channels=self.output_channels,
            output_image_size=(self.img_size, self.img_size),
            decoder_dropout=0.0,
            encoder_type='mlp',
            encoder_params={'num_hidden_layers': 1, 'hidden_dim': 20, 'activation_fn_str': 'relu', 'dropout_rate': 0.0},
            decoder_patch_size=self.decoder_patch_size
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()
        target = torch.randn(self.batch_size, self.output_channels, self.img_size, self.img_size)
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            out = model(self.dummy_img, self.dummy_action)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    def test_reward_predictor_mlp_compatibility(self):
        # Test that reward predictor MLP can be used with the encoder output
        model = EncoderDecoderJEPAStyle(
            image_size=self.img_size,
            patch_size=self.decoder_patch_size,
            input_channels=self.input_channels,
            action_dim=self.action_dim,
            action_emb_dim=self.action_emb_dim,
            latent_dim=self.latent_dim,
            predictor_hidden_dim=self.predictor_hidden_dim,
            predictor_output_dim=self.decoder_dim,
            predictor_dropout_rate=0.0,
            decoder_dim=self.decoder_dim,
            decoder_depth=self.decoder_depth,
            decoder_heads=self.decoder_heads,
            decoder_mlp_dim=self.decoder_mlp_dim,
            output_channels=self.output_channels,
            output_image_size=(self.img_size, self.img_size),
            decoder_dropout=0.0,
            encoder_type='mlp',
            encoder_params={'num_hidden_layers': 1, 'hidden_dim': 20, 'activation_fn_str': 'relu', 'dropout_rate': 0.0},
            decoder_patch_size=self.decoder_patch_size
        ).to(self.device)
        # Get encoder output
        with torch.no_grad():
            latent = model.encoder(self.dummy_img)
        reward_mlp = RewardPredictorMLP(
            input_dim=latent.shape[1],
            hidden_dims=[8, 4],
            activation_fn_str='relu',
            use_batch_norm=False,
            dropout_rate=0.0
        ).to(self.device)
        reward = reward_mlp(latent)
        self.assertEqual(reward.shape, (self.batch_size, 1))

if __name__ == '__main__':
    unittest.main() 