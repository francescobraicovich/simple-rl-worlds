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
