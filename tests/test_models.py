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
        self.dummy_img_square = torch.randn(
            self.batch_size, self.input_channels, self.image_size_int, self.image_size_int)
        self.dummy_img_rect = torch.randn(
            self.batch_size, self.input_channels, self.image_size_tuple[0], self.image_size_tuple[1])

    def test_cnn_encoder_initialization_and_forward(self):
        # Test with square image
        cnn_encoder = CNNEncoder(input_channels=self.input_channels,
                                 image_size=self.image_size_int, latent_dim=self.latent_dim)
        output = cnn_encoder(self.dummy_img_square)
        self.assertEqual(output.shape, (self.batch_size, self.latent_dim))

        # Test with rectangular image
        cnn_encoder_rect = CNNEncoder(input_channels=self.input_channels,
                                      image_size=self.image_size_tuple, latent_dim=self.latent_dim)
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
        self.assertEqual(output_custom.shape,
                         (self.batch_size, self.latent_dim))

    def test_mlp_encoder_initialization_and_forward(self):
        # Test with square image
        mlp_encoder = MLPEncoder(input_channels=self.input_channels,
                                 image_size=self.image_size_int, latent_dim=self.latent_dim)
        output = mlp_encoder(self.dummy_img_square)
        self.assertEqual(output.shape, (self.batch_size, self.latent_dim))

        # Test with rectangular image
        mlp_encoder_rect = MLPEncoder(input_channels=self.input_channels,
                                      image_size=self.image_size_tuple, latent_dim=self.latent_dim)
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
        self.assertEqual(output_custom.shape,
                         (self.batch_size, self.latent_dim))


class TestModelSelection(unittest.TestCase):
    def setUp(self):
        self.image_size = 64
        self.patch_size = 16  # For ViT and decoder
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

        self.dummy_img = torch.randn(
            2, self.input_channels, self.image_size, self.image_size)
        # Assuming continuous or already processed action
        self.dummy_action = torch.randn(2, self.action_dim)

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
        model_vit(self.dummy_img, self.dummy_action)  # Check forward pass

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
        jepa_vit(self.dummy_img, self.dummy_action,
                 self.dummy_img)  # Check forward pass

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


class TestJEPATargetEncoderModes(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 32  # Smaller for faster tests
        self.patch_size = 8
        self.input_channels = 1 # Simpler
        self.action_dim = 2
        self.action_emb_dim = 4
        self.latent_dim = 8 # Predictor output dim must match latent_dim
        self.predictor_hidden_dim = 16
        self.ema_decay = 0.5 # For easier testing of weight changes
        self.encoder_type = 'cnn'
        self.encoder_params = {
            'num_conv_layers': 1,
            'base_filters': 4,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'fc_hidden_dim': None # Ensure direct output to latent_dim
        }

        self.s_t = torch.randn(self.batch_size, self.input_channels, self.image_size, self.image_size)
        self.action = torch.randn(self.batch_size, self.action_dim)
        self.s_t_plus_1 = torch.randn(self.batch_size, self.input_channels, self.image_size, self.image_size)

    def _create_jepa_model(self, target_encoder_mode):
        return JEPA(
            image_size=self.image_size,
            patch_size=self.patch_size,
            input_channels=self.input_channels,
            action_dim=self.action_dim,
            action_emb_dim=self.action_emb_dim,
            latent_dim=self.latent_dim,
            predictor_hidden_dim=self.predictor_hidden_dim,
            predictor_output_dim=self.latent_dim, # Must match latent_dim
            ema_decay=self.ema_decay,
            encoder_type=self.encoder_type,
            encoder_params=self.encoder_params,
            target_encoder_mode=target_encoder_mode
        )

    def _get_target_encoder_weights(self, model):
        if model.target_encoder is not None:
            return {k: v.clone() for k, v in model.target_encoder.state_dict().items()}
        return None

    def _check_weights_equal(self, weights1_dict, weights2_dict):
        if weights1_dict is None and weights2_dict is None:
            return True
        if weights1_dict is None or weights2_dict is None:
            return False
        if len(weights1_dict) != len(weights2_dict):
            return False
        for key in weights1_dict:
            if not torch.equal(weights1_dict[key], weights2_dict[key]):
                return False
        return True

    def _perform_gradient_check(self, model, outputs, target_encoder_expected_grad=False):
        # Dummy loss: sum of the first output tensor (predicted_s_t_plus_1_embedding)
        loss = outputs[0].sum()
        model.zero_grad() # Zero gradients before backward pass
        loss.backward()

        # Online encoder should always have gradients
        for param in model.online_encoder.parameters():
            self.assertIsNotNone(param.grad, "Online encoder should have gradients.")
            self.assertTrue(param.grad.abs().sum() > 0, "Online encoder gradient sum should be > 0")


        # Predictor should always have gradients
        for param in model.predictor.parameters():
            self.assertIsNotNone(param.grad, "Predictor should have gradients.")
            self.assertTrue(param.grad.abs().sum() > 0, "Predictor gradient sum should be > 0")


        # Target encoder (if exists) should not have gradients
        if model.target_encoder is not None:
            for param in model.target_encoder.parameters():
                if target_encoder_expected_grad: # This case should ideally not happen for target encoder
                     self.assertIsNotNone(param.grad, "Target encoder was expected to have grad but did not.")
                else:
                     self.assertIsNone(param.grad, "Target encoder should not have gradients.")
        model.zero_grad() # Clean up gradients

    def test_mode_default(self):
        mode = "default"
        model = self._create_jepa_model(mode)
        self.assertIsNotNone(model.target_encoder, f"Target encoder should exist for mode '{mode}'")

        # Check requires_grad
        for param in model.online_encoder.parameters(): self.assertTrue(param.requires_grad)
        for param in model.predictor.parameters(): self.assertTrue(param.requires_grad)
        for param in model.target_encoder.parameters(): self.assertFalse(param.requires_grad)

        initial_target_weights = self._get_target_encoder_weights(model)

        outputs = model(self.s_t, self.action, self.s_t_plus_1)
        pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = outputs

        self.assertEqual(pred_emb.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(target_emb_detached.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(online_s_t_emb.shape, (self.batch_size, self.latent_dim))
        self.assertIsNotNone(online_s_t_plus_1_emb, f"online_s_t_plus_1_emb should not be None for mode '{mode}'")
        self.assertEqual(online_s_t_plus_1_emb.shape, (self.batch_size, self.latent_dim))

        # Target weights should not change during forward for 'default'
        current_target_weights_after_fwd = self._get_target_encoder_weights(model)
        self.assertTrue(self._check_weights_equal(initial_target_weights, current_target_weights_after_fwd),
                        "Target encoder weights should not change during forward pass for mode 'default'.")

        model.perform_ema_update()
        current_target_weights_after_ema = self._get_target_encoder_weights(model)
        self.assertFalse(self._check_weights_equal(initial_target_weights, current_target_weights_after_ema),
                         "Target encoder weights should change after perform_ema_update for mode 'default'.")

        self._perform_gradient_check(model, outputs)


    def test_mode_vjepa2(self):
        mode = "vjepa2"
        model = self._create_jepa_model(mode)
        self.assertIsNotNone(model.target_encoder, f"Target encoder should exist for mode '{mode}'")

        # Check requires_grad
        for param in model.online_encoder.parameters(): self.assertTrue(param.requires_grad)
        for param in model.predictor.parameters(): self.assertTrue(param.requires_grad)
        for param in model.target_encoder.parameters(): self.assertFalse(param.requires_grad)

        initial_target_weights = self._get_target_encoder_weights(model)

        outputs = model(self.s_t, self.action, self.s_t_plus_1)
        pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = outputs

        self.assertEqual(pred_emb.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(target_emb_detached.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(online_s_t_emb.shape, (self.batch_size, self.latent_dim))
        self.assertIsNone(online_s_t_plus_1_emb, f"online_s_t_plus_1_emb should be None for mode '{mode}'")

        current_target_weights_after_fwd = self._get_target_encoder_weights(model)
        self.assertFalse(self._check_weights_equal(initial_target_weights, current_target_weights_after_fwd),
                         "Target encoder weights should change during forward pass for mode 'vjepa2'.")

        # perform_ema_update should do nothing for vjepa2
        model.perform_ema_update()
        current_target_weights_after_ema_call = self._get_target_encoder_weights(model)
        self.assertTrue(self._check_weights_equal(current_target_weights_after_fwd, current_target_weights_after_ema_call),
                        "Target encoder weights should NOT change after perform_ema_update for mode 'vjepa2'.")

        self._perform_gradient_check(model, outputs)

    def test_mode_none(self):
        mode = "none"
        model = self._create_jepa_model(mode)
        self.assertIsNone(model.target_encoder, f"Target encoder should be None for mode '{mode}'")

        # Check requires_grad
        for param in model.online_encoder.parameters(): self.assertTrue(param.requires_grad)
        for param in model.predictor.parameters(): self.assertTrue(param.requires_grad)

        outputs = model(self.s_t, self.action, self.s_t_plus_1)
        pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = outputs

        self.assertEqual(pred_emb.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(target_emb_detached.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(online_s_t_emb.shape, (self.batch_size, self.latent_dim))
        self.assertIsNotNone(online_s_t_plus_1_emb, f"online_s_t_plus_1_emb should not be None for mode '{mode}'")
        self.assertEqual(online_s_t_plus_1_emb.shape, (self.batch_size, self.latent_dim))

        # perform_ema_update should do nothing (no target encoder)
        model.perform_ema_update()
        # No specific weight check needed as target_encoder is None.
        # Just ensure no error occurs.

        self._perform_gradient_check(model, outputs)


if __name__ == '__main__':
    unittest.main()
