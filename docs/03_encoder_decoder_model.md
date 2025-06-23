# 03 Encoder-Decoder Model Architecture

The Standard Encoder-Decoder model, implemented in `src/models/encoder_decoder.py`, represents a common approach for learning world models by directly predicting future states in pixel space. Given the current state (an image `s_t`) and an action `a_t`, the model aims to generate the next state image `s_t+1`.

<!-- TODO: Add an architectural diagram here -->
```
s_t (image) --> [ENCODER] --+--> latent_s_t --+
                            |                  |
a_t (vector) -> [ACTION_EMB]-+-> embedded_a_t --+--> [DECODER_INPUT_PROJ] --> decoder_memory --> [TRANSFORMER_DECODER with query_tokens] --> decoded_patches --> [TO_PIXELS] --> predicted_s_t+1 (image)
```
*(Simplified ASCII diagram)*

## Model Components

The architecture consists of three main components: an encoder, an action embedding layer, and a decoder.

### 1. Encoder

The encoder is responsible for processing the input state image `s_t` and mapping it to a lower-dimensional latent representation, `latent_s_t`. This compressed representation aims to capture the salient features of the input. Our framework supports several types of encoders, configurable via the `encoder_type` parameter in `config.yaml`:

-   **Vision Transformer (ViT):** (`encoder_type: "vit"`)
    The ViT processes images by dividing them into fixed-size patches, linearly embedding these patches, adding positional embeddings, and then feeding them into a standard Transformer encoder. Key parameters (specified in `config.yaml` under `encoder_params.vit`) include `patch_size`, `depth` (number of Transformer blocks), `heads` (number of attention heads), and `mlp_dim`. The output can be taken from the special `[CLS]` token or by mean pooling over the patch embeddings, resulting in a `latent_dim` sized vector.

-   **Convolutional Neural Network (CNN):** (`encoder_type: "cnn"`)
    The CNN encoder, implemented in `src/models/cnn.py`, uses a series of convolutional layers to extract spatial hierarchies of features. Parameters (under `encoder_params.cnn`) include `num_conv_layers`, `base_filters`, `kernel_size`, `stride`, and `activation_fn_str`. The output of the convolutional stack is typically flattened and passed through an optional fully connected layer to produce the `latent_dim` sized `latent_s_t`.

-   **Multi-Layer Perceptron (MLP):** (`encoder_type: "mlp"`)
    For simpler or non-image-based environments, an MLP encoder (`src/models/mlp.py`) can be used. It flattens the input image and processes it through several fully connected layers. Parameters (under `encoder_params.mlp`) include `num_hidden_layers`, `hidden_dim`, and `activation_fn_str`.

The choice of encoder and its configuration (e.g., `latent_dim`, `patch_size`, specific layer parameters) are crucial for determining the nature and quality of the learned state representation.

### 2. Action Embedding

The action `a_t` taken by the agent is typically a discrete or continuous vector. To integrate this action into the prediction process, it is first embedded into a higher-dimensional space using a simple linear layer (`self.action_embedding` in `StandardEncoderDecoder`). The dimensionality of this embedding is specified by `action_emb_dim` in `config.yaml`. This `embedded_action` provides the decoder with information about the agent's intervention.

### 3. Transformer-Based Decoder

The core of the predictive component is a Transformer-based decoder. Its goal is to generate the next state image `s_t+1` based on the encoded current state `latent_s_t` and the `embedded_action`.

The process involves several steps:
1.  **Input Projection:** The `latent_s_t` and `embedded_action` are concatenated and then linearly projected by `self.decoder_input_projection` to match the `decoder_dim`, which is the internal dimension of the Transformer decoder. This projected vector serves as the `memory` for the decoder.
2.  **Query Tokens:** A set of learnable query tokens (`self.decoder_query_tokens`) are initialized. These tokens act as the initial input (`tgt`) to the Transformer decoder and their number corresponds to the number of patches in the output image.
3.  **Transformer Decoder Layers:** The `self.transformer_decoder` (an instance of `nn.TransformerDecoder`) processes these query tokens, attending to the `memory` (combined state-action representation). This allows the model to synthesize information from the current state and action to predict the features of each patch in the next state. Key parameters for the decoder, configured in `config.yaml`, include `decoder_dim`, `decoder_depth` (number of layers), `decoder_heads` (attention heads), and `decoder_mlp_dim`.
4.  **Pixel Projection:** The output of the Transformer decoder, which represents the predicted features for each output patch, is passed through a linear layer (`self.to_pixels`). This layer projects the features into a dimension corresponding to the flattened pixel values of a patch (`output_channels * decoder_patch_size * decoder_patch_size`).
5.  **Image Reconstruction:** Finally, the predicted patches are rearranged from a sequence of flattened patches back into a complete image (`predicted_s_t+1`) using an `einops.Rearrange` operation (`self.patch_to_image`). The `decoder_patch_size` determines the granularity of this reconstruction.

## Training Process

The Standard Encoder-Decoder model is trained within a supervised learning paradigm. The goal is to minimize the difference between its predicted next state and the actual next state observed from the environment.

The training process is orchestrated by the `src/training_engine.py`. This engine utilizes a dedicated training and validation loop, primarily implemented in `src/training_loops/epoch_loop.py` (specifically, the `train_eval_epoch_enc_dec` function or similar, tailored for this model type). This loop typically handles:
-   Iterating through the training dataset in batches.
-   Performing a forward pass of the model with the current batch of `(state, action, next_state)` tuples to get `predicted_next_state`.
-   Calculating the loss (e.g., MSE) between `predicted_next_state` and `actual_next_state`.
-   Performing backpropagation to compute gradients.
-   Taking a step with the optimizer to update the model's weights.
-   Periodically running a validation loop on a separate dataset to monitor performance on unseen data and check for early stopping conditions.

For more detailed information on how to configure and run the training, including setting up the environment, data, and specific hyperparameters, please refer to the **[`docs/06_usage_guide.md`](../06_usage_guide.md)**.

## Loss Function

The Standard Encoder-Decoder model is typically trained using a direct pixel-wise comparison between the predicted next state image `\hat{s}_{t+1}` and the ground truth next state image `s_{t+1}`. The most common loss function for this purpose is the **Mean Squared Error (MSE)**:

`L_MSE = (1/N) * Σ || \hat{s}_{t+1} - s_{t+1} ||^2`

where `N` is the total number of pixels in the image. This loss function penalizes deviations in pixel intensity, encouraging the decoder to produce images that are as close as possible to the actual observed next states.

## Discussion

### Advantages:
-   **Directness and Interpretability:** The model directly predicts the full next state in pixel space, making the output easily interpretable as an image. This can be useful for visualization and debugging.
-   **Strong Supervision Signal:** Pixel-level reconstruction provides a dense and strong supervisory signal, which can facilitate learning in some contexts.

### Disadvantages:
-   **Computational Cost:** Predicting every pixel in a high-resolution image can be computationally very expensive, both in terms of model parameters and processing time. The Transformer decoder, in particular, can be demanding.
-   **Focus on Pixel-Level Details:** By minimizing pixel-wise errors, the model might expend significant capacity on accurately reconstructing fine-grained visual details that may be irrelevant for understanding the environment's dynamics or for downstream control tasks. This can sometimes hinder the learning of more abstract or semantically meaningful representations.
-   **Difficulty with Stochasticity:** Pixel-space prediction can struggle in highly stochastic environments where many plausible next states could follow a given state-action pair. Averaging over these possibilities can lead to blurry or unrealistic predictions.

The Standard Encoder-Decoder model serves as a fundamental baseline for world models. Its strengths in direct prediction are balanced by potential drawbacks related to computational load and the nature of the learned representations. These aspects motivate the exploration of alternative approaches, such as those that predict in a more abstract latent space, like the JEPA model discussed next.

## Relevant Configuration Parameters

The behavior, architecture, and training of the Standard Encoder-Decoder model are controlled by various parameters defined in the `config.yaml` file. Below are some of the key parameters relevant to this model:

-   **`encoder_type`**: (string, e.g., `"vit"`, `"cnn"`, `"mlp"`) Chooses the architecture for the encoder module.
-   **`encoder_params`**: (dict) A nested dictionary containing parameters specific to the chosen `encoder_type`. For example:
    -   `encoder_params.vit`: Contains settings like `patch_size`, `depth`, `heads`, `mlp_dim` for the Vision Transformer encoder.
    -   `encoder_params.cnn`: Contains settings like `num_conv_layers`, `base_filters`, `kernel_size` for the CNN encoder.
    -   `encoder_params.mlp`: Contains settings like `num_hidden_layers`, `hidden_dim` for the MLP encoder.
-   **`latent_dim`**: (integer) The dimensionality of the latent representation `latent_s_t` produced by the encoder.
-   **`action_emb_dim`**: (integer) The dimensionality of the embedding for the action `a_t`.
-   **`decoder_dim`**: (integer) The internal dimensionality of the Transformer decoder (often referred to as `d_model`).
-   **`decoder_depth`**: (integer) The number of layers in the Transformer decoder.
-   **`decoder_heads`**: (integer) The number of attention heads in the Transformer decoder's multi-head attention mechanisms.
-   **`decoder_mlp_dim`**: (integer) The dimensionality of the feed-forward network within each Transformer decoder block.
-   **`decoder_patch_size`**: (integer) The size of the patches used by the decoder to reconstruct the output image. The output image dimensions must be divisible by this patch size.
-   **`decoder_dropout`**: (float) Dropout rate applied within the Transformer decoder.
-   **`learning_rate`**: (float) The learning rate for the Adam optimizer (or other chosen optimizer) used for training the model.
-   **`batch_size`**: (integer) The number of samples processed in each training iteration.
-   **`num_epochs`**: (integer) The maximum number of epochs to train the model.
-   **`early_stopping.metric_enc_dec`**: (string, e.g., `"val_loss_enc_dec"`) The metric monitored on the validation set to make early stopping decisions for this model.
-   **`early_stopping.checkpoint_path_enc_dec`**: (string) Filename used to save the best performing model checkpoint based on the early stopping metric.
-   **`training_options.skip_std_enc_dec_training_if_loaded`**: (boolean) If `true`, and a pre-trained Standard Encoder-Decoder model is successfully loaded (via `model_type_to_load` and `load_model_path`), the training phase for this model will be skipped.

For a comprehensive list of all configuration options and their detailed explanations, please refer directly to the `config.yaml` file (which is heavily commented) and the **[`docs/06_usage_guide.md`](../06_usage_guide.md)**.

## Encoder-Decoder JEPA-Style Variant (Fair JEPA Baseline)

To enable rigorous, apples-to-apples comparison with JEPA, this repository includes an **Encoder-Decoder JEPA-style** model variant. This model:

- Uses the same encoder options (ViT, CNN, MLP) as the standard Encoder-Decoder and JEPA.
- After encoding the current state, concatenates the action embedding (as in both current models).
- Passes the concatenated vector through a JEPA-style predictor MLP (identical to the one used in JEPA).
- Feeds the predictor's output into a Transformer-based decoder (as in the standard Encoder-Decoder and JEPAStateDecoder) to reconstruct the next state image in pixel space.
- All architectural parameters (encoder, predictor, decoder) are fully configurable via `config.yaml`.

**Purpose:**
This model is designed to match the parameter count and architectural complexity of JEPA as closely as possible, isolating the effect of the core modeling principle (predicting in pixel space vs. embedding space). By comparing this model to both the standard Encoder-Decoder and JEPA, researchers can determine whether JEPA's performance gains are due to its architectural philosophy or simply increased model capacity.

**Usage:**
- Select this model by setting `model_type_to_load: "enc_dec_jepa_style"` in `config.yaml`.
- Configure all relevant parameters (encoder, predictor, decoder) as for the other models.
- The model can be trained and evaluated using the same training and reward prediction pipelines as the other world models.
