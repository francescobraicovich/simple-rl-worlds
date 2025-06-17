# 04 Joint Embedding Predictive Architecture (JEPA) Model

The Joint Embedding Predictive Architecture (JEPA), as implemented in `src/models/jepa.py`, offers an alternative to direct pixel-space prediction by operating entirely within an abstract embedding space. This approach is designed to encourage the learning of more semantically meaningful and efficient representations by abstracting away irrelevant pixel-level details. The core idea is to predict the embedding of a future state `s_t+1` given an embedding of the current state `s_t` and the action `a_t`.

## Core JEPA Components

The JEPA model comprises three principal components: an online encoder, a target encoder, and a predictor.

### 1. Online Encoder

-   **Role:** The online encoder is responsible for mapping the input images (current state `s_t` and next state `s_t+1`) into fixed-dimensional embeddings: `online_encoded_s_t` (denoted as `z_t`) and `online_encoded_s_t_plus_1` (denoted as `z_{t+1}`).
-   **Training:** This encoder is trained via standard backpropagation. Its parameters are updated by the gradients originating from both the primary prediction loss (indirectly, as the predictor aims to match targets derived from the target encoder, which in turn is related to the online encoder) and, crucially, the auxiliary losses designed to shape its output embedding space.
-   **Configurability:** Similar to the Standard Encoder-Decoder model, the online encoder can be a Vision Transformer (ViT), a Convolutional Neural Network (CNN), or a Multi-Layer Perceptron (MLP). The choice is determined by the `encoder_type` parameter in `config.yaml`, with specific configurations detailed under `encoder_params`. The output dimension is `latent_dim`.

### 2. Target Encoder

-   **Architecture:** The target encoder has an identical architecture to the online encoder. It is initialized as a deep copy of the online encoder.
-   **Weight Updates:** Crucially, the target encoder's weights are **not** updated via backpropagation. Instead, they are updated using an Exponential Moving Average (EMA) of the online encoder's weights. The update rule is:
    `θ_target = ema_decay * θ_target + (1 - ema_decay) * θ_online`
    where `ema_decay` is a hyperparameter (e.g., 0.996, configured as `jepa_model.ema_decay` in `config.yaml`). This update is performed by the `_update_target_encoder_ema()` method in `JEPA` class.
-   **Purpose:** The target encoder provides stable and slowly evolving target embeddings (`target_encoded_s_t` or `z'_t`, and `target_encoded_s_t_plus_1` or `z'_{t+1}`). This stability is essential for the predictor, as it prevents the common issue in siamese networks where the predictor could chase rapidly changing targets produced by an online encoder, potentially leading to trivial solutions or training instability. The `target_encoded_s_t_plus_1` serves as the direct regression target for the predictor.

### 3. Predictor

-   **Architecture:** The predictor is typically an MLP (as implemented in `src/models/jepa.py` within the `JEPA` class) but could also be a Transformer or other sequence processing model.
-   **Function:** It takes the target-encoded representation of the current state (`target_encoded_s_t`, i.e., `z'_t`) and an embedding of the action `a_t` (produced by `self.action_embedding`) as input. Its objective is to predict the embedding of the *next state*, specifically aiming to match the *target encoder's* representation of `s_t+1`. That is, it outputs `predicted_s_t_plus_1_embedding` (denoted `\hat{z}_{t+1}`), which should be close to `target_encoded_s_t_plus_1` (`z'_{t+1}`).
-   **Configuration:** Key parameters for the predictor MLP, such as `jepa_predictor_hidden_dim` and `predictor_output_dim` (which must equal `latent_dim`), are set in `config.yaml`.

### 4. Target Encoder Mode (`jepa_model.target_encoder_mode`)

The `jepa_model.target_encoder_mode` parameter in `config.yaml` dictates the operational strategy for the online and target encoders. This choice significantly influences how predictions are made, how targets are generated, and how EMA updates and auxiliary losses are handled. It offers three distinct modes:

-   **`"default"`**:
    -   **Description:** This mode implements a standard JEPA behavior.
    -   **Predictor Input (`s_t` representation):** Provided by the **target encoder** (`target_encoder(s_t)`).
    -   **Prediction Target (`s_{t+1}` representation):** Provided by the **target encoder** (`target_encoder(s_{t+1}).detach()`).
    -   **EMA Updates:** The target encoder's weights are updated via EMA of the online encoder's weights after each training step.
    -   **Auxiliary Loss Input:** Typically uses online encoder representations of both `s_t` (`online_encoder(s_t)`) and `s_{t+1}` (`online_encoder(s_{t+1})`).

-   **`"vjepa2"`**:
    -   **Description:** Inspired by V-JEPA, this mode uses the online encoder for the current state representation fed to the predictor.
    -   **Predictor Input (`s_t` representation):** Provided by the **online encoder** (`online_encoder(s_t)`).
    -   **Prediction Target (`s_{t+1}` representation):** Provided by the **target encoder** (`target_encoder(s_{t+1}).detach()`).
    -   **EMA Updates:** The target encoder is updated via EMA of the online encoder's weights.
    -   **Auxiliary Loss Input:** Typically uses only the online encoder's representation of the current state `s_t` (`online_encoder(s_t)`). The `forward` method of the JEPA model might return `None` for the `s_{t+1}` online embedding in this mode to signal this.

-   **`"none"`**:
    -   **Description:** This mode disables the target encoder. All representations for prediction and targets come from the online encoder.
    -   **Predictor Input (`s_t` representation):** Provided by the **online encoder** (`online_encoder(s_t)`).
    -   **Prediction Target (`s_{t+1}` representation):** Provided by the **online encoder** (`online_encoder(s_{t+1}).detach()`).
    -   **EMA Updates:** No EMA updates are performed as the target encoder is not used.
    -   **Auxiliary Loss Input:** Uses online encoder representations of both `s_t` (`online_encoder(s_t)`) and `s_{t+1}` (`online_encoder(s_{t+1})`).

The choice of `target_encoder_mode` is critical for defining the specific JEPA variant being experimented with and impacts the learned representations and model stability.

## Training Process

The JEPA model (comprising the online encoder and the predictor) is trained using a supervised learning approach, where the "supervision" comes from predicting target embeddings generated by the (potentially EMA-updated) target encoder.

### JEPA Model Training (Online Encoder & Predictor)

The primary training process for the JEPA model is managed by `src/training_engine.py`. This engine employs a versatile training and validation loop, typically found within `src/training_loops/epoch_loop.py` (e.g., the `train_validate_model_epoch` function, which is adapted based on the model type, or a JEPA-specific derivative). This loop generally performs the following steps for each batch:

1.  **Forward Pass:**
    *   The online encoder processes the current state `s_t` and next state `s_{t+1}` to produce `online_encoded_s_t` and `online_encoded_s_{t+1}`.
    *   The target encoder (if active, depending on `target_encoder_mode`) processes `s_t` and `s_{t+1}` to produce `target_encoded_s_t` and `target_encoded_s_{t+1}`.
    *   The predictor takes the appropriate representation of `s_t` (from online or target encoder based on `target_encoder_mode`) and the embedded action `a_t` to predict the embedding of `s_{t+1}`.
2.  **Loss Calculation:**
    *   **Prediction Loss:** Calculated between the predictor's output and the (detached) target representation of `s_{t+1}` (from online or target encoder).
    *   **Auxiliary Loss:** Calculated based on the outputs of the online encoder, using the configured auxiliary loss type (e.g., VICReg, Barlow Twins).
    *   The total loss is a weighted sum of the prediction and auxiliary losses.
3.  **Backpropagation:** Gradients are computed for the online encoder and predictor based on the total loss.
4.  **Optimizer Step:** The weights of the online encoder and predictor are updated.
5.  **EMA Update:** If a target encoder is used (`"default"` or `"vjepa2"` modes), its weights are updated via Exponential Moving Average (EMA) of the online encoder's weights.
6.  **Validation:** Periodically, the model is evaluated on a validation set to monitor performance and check for early stopping conditions.

### JEPAStateDecoder Training (Optional)

If the `JEPAStateDecoder` is enabled (via `jepa_decoder_training.enabled: true` in `config.yaml`), it is trained in a separate, subsequent phase after the main JEPA model training (or loading) is complete. This decoder is designed to reconstruct the pixel-space state `s_t` from the learned JEPA embeddings (`online_encoded_s_t` or `target_encoded_s_t`, depending on configuration).

-   **Purpose:** The `JEPAStateDecoder` serves as a tool for visually inspecting and evaluating the quality of the representations learned by the JEPA model. It is not part of the core JEPA representation learning process itself.
-   **Training Management:** Its training is also orchestrated by `src/training_engine.py`, which calls the `train_jepa_state_decoder` function located in `src/training_loops/jepa_decoder_loop.py`. This loop handles the specific training requirements for the decoder, such as loading JEPA embeddings and optimizing the decoder for image reconstruction.

For practical details on configuring and running the training for both the JEPA model and its optional state decoder, please refer to **[`docs/06_usage_guide.md`](../06_usage_guide.md)**.

## Prediction Loss

The primary training signal for the predictor comes from minimizing the discrepancy between its predicted embedding of the next state and the actual target-encoded embedding of the next state.

-   **Calculation:** This loss is typically a Mean Squared Error (MSE) computed as:
    `L_pred = MSE(predicted_s_t_plus_1_embedding, target_encoded_s_t_plus_1.detach())`
    The `.detach()` is crucial as the target encoder's output serves as a fixed target and should not propagate gradients back to the target encoder itself.
-   **Embedding Space Prediction:** A key characteristic of JEPA is that this prediction and loss calculation occur entirely in the *embedding space*, not pixel space.
    -   **Potential Benefits:**
        -   **Abstract Representations:** By predicting in embedding space, the model is encouraged to learn representations that capture abstract, high-level concepts about state transitions rather than focusing on pixel-level details. This can lead to more robust and generalizable world models.
        -   **Computational Efficiency:** Predicting a lower-dimensional embedding is significantly less computationally intensive than reconstructing an entire high-dimensional image.
        -   **Focus on Dynamics:** The model can concentrate its capacity on understanding the dynamics of how states change under actions, rather than on visual reconstruction.

## Auxiliary Losses for Representation Learning

A critical aspect of training JEPA models effectively is the use of auxiliary losses applied directly to the outputs of the **online encoder** (`online_encoded_s_t` and `online_encoded_s_t_plus_1`). These losses are essential to prevent representational collapse (where the encoder outputs trivial or constant embeddings) and to encourage the learning of informative, well-structured, and diverse embeddings. Without them, the online encoder might learn to produce outputs that are trivial for the predictor to match, without capturing useful information.

Our framework implements several such auxiliary losses, configurable via the `auxiliary_loss` section in `config.yaml`. The chosen loss function's `calculate_reg_terms` method is typically called on `online_encoded_s_t` and/or `online_encoded_s_t_plus_1`. The `auxiliary_loss.weight` parameter controls the contribution of this loss to the total JEPA loss.

### 1. VICReg (Variance-Invariance-Covariance Regularization)

-   **Reference:** `src/losses/vicreg.py`
-   **Objective:** VICReg aims to learn informative representations by simultaneously:
    1.  Maintaining variance in the embeddings (Variance Term).
    2.  Making embeddings from different views of the same sample similar (Invariance Term - adapted for JEPA).
    3.  Decorrelating different dimensions of the embeddings (Covariance Term).
-   **Components in our JEPA Context:** The `VICRegLoss.calculate_reg_terms(z)` method is used, which applies the variance and covariance terms to a single batch of embeddings `z` (e.g., `online_encoded_s_t`).
    -   **Variance Term (`std_coeff`):** This term encourages the standard deviation of each dimension of the batch of embeddings to be close to a target value (typically 1). It uses a hinge loss (`F.relu(1 - z_std)`) to penalize standard deviations below this target, promoting the utilization of the entire embedding space and preventing features from collapsing to zero variance.
    -   **Covariance Term (`cov_coeff`):** This term aims to decorrelate the different dimensions of the learned embeddings. It penalizes the sum of squared off-diagonal elements of the covariance matrix of the embeddings. Minimizing this term encourages each dimension to carry unique information.
    -   **Invariance Term (`sim_coeff`):** In the original VICReg paper, the `sim_coeff` applies to an MSE loss between embeddings of two augmented views of the same image (`F.mse_loss(x,y)`). In our JEPA setup, when `calculate_reg_terms(z)` is used on a single embedding `z`, this specific term is not directly computed as there's no second view `y` in that call. The `config.yaml` default sets `sim_coeff: 0.0` for `auxiliary_loss.params.vicreg` when used this way, effectively focusing on the variance and covariance regularization terms. If the full `forward(x,y)` method were used (e.g., by passing `online_encoded_s_t` and `online_encoded_s_t_plus_1` as x and y, though this is not the typical JEPA setup for VICReg), then this term would represent the similarity between these two sequential embeddings.

### 2. Barlow Twins

-   **Reference:** `src/losses/barlow_twins.py`
-   **Objective:** Barlow Twins encourages the cross-correlation matrix computed from the online encoder's output embeddings (for a batch) to be as close to the identity matrix as possible.
-   **Mechanism:**
    1.  Embeddings `z` (e.g., `online_encoded_s_t`) are first standardized along the batch dimension (mean 0, std 1 for each feature).
    2.  The auto-correlation matrix `C = (z_norm.T @ z_norm) / batch_size` is computed.
    3.  The loss function then has two parts:
        -   **Invariance Term:** Penalizes the diagonal elements of `C` for deviating from 1. This encourages each feature to have unit variance after normalization.
        -   **Redundancy Reduction Term:** Penalizes the off-diagonal elements of `C` for being non-zero. This encourages different features to be decorrelated. The `lambda_param` (configured in `config.yaml` under `auxiliary_loss.params.barlow_twins`) weights this term.
-   The `BarlowTwinsLoss.calculate_reg_terms(z)` method computes this for a single batch of embeddings.

### 3. DINO (Centering Component)

-   **Reference:** `src/losses/dino.py`
-   **Objective:** This implementation focuses on the centering mechanism inspired by DINO, which helps prevent model collapse, particularly where all outputs converge to a zero vector.
-   **Mechanism:**
    1.  An EMA-updated `center` vector (of dimension `latent_dim`) is maintained.
    2.  The loss penalizes the squared Euclidean distance between the mean of the current batch of online embeddings (`online_encoded_s_t` or `online_encoded_s_t_plus_1`) and this `center`.
    3.  After computing the loss with the current `center`, the `center` itself is updated using an EMA of the current batch's mean.
    `center = center_ema_decay * center + (1 - center_ema_decay) * batch_mean`
-   The `center_ema_decay` is configurable in `config.yaml` under `auxiliary_loss.params.dino`. The `DINOLoss.calculate_reg_terms(z)` method provides this loss.

### Configuration and Application

The choice of auxiliary loss (`type`) and its specific parameters (`params`) are set in the `auxiliary_loss` section of `config.yaml`. The overall weight of this loss component is controlled by `auxiliary_loss.weight`. These losses are applied to the outputs of the online encoder (e.g., `online_encoded_s_t` and sometimes `online_encoded_s_t_plus_1`, depending on the training script's specific implementation details for applying the auxiliary loss). This ensures that the representations `z_t` and `z_{t+1}` are "well-behaved" before being implicitly used in the prediction task.

## Overall JEPA Training Objective

The total loss function for training the JEPA model (specifically the online encoder and the predictor) is a weighted sum of the primary prediction loss and the chosen auxiliary regularization loss:

`L_total_JEPA = L_pred + auxiliary_loss.weight * L_aux`

This combined objective trains the predictor to make accurate forecasts in the embedding space while simultaneously ensuring that the online encoder learns rich, non-collapsed representations. The target encoder provides the stable targets necessary for this learning process.

By operating in embedding space and carefully regularizing its representations, JEPA aims to learn world models that are both efficient and effective at capturing the underlying dynamics of an environment.

## Relevant Configuration Parameters

The architecture, training, and behavior of the JEPA model and its optional `JEPAStateDecoder` are controlled by parameters in `config.yaml`.

### Core JEPA Model Parameters:

-   **`encoder_type`**: (string, e.g., `"vit"`, `"cnn"`, `"mlp"`) Shared encoder architecture for online and target encoders.
-   **`encoder_params`**: (dict) Nested dictionary with parameters for the chosen `encoder_type` (e.g., `encoder_params.vit`).
-   **`latent_dim`**: (integer) Dimensionality of the embeddings produced by the encoders and targeted by the predictor.
-   **`action_emb_dim`**: (integer, top-level) Dimensionality of the action embedding used by the predictor.
-   **`jepa_model.predictor_hidden_dim`**: (integer) Hidden dimension size within the MLP predictor.
-   **`jepa_model.predictor_dropout_rate`**: (float) Dropout rate in the predictor.
-   **`jepa_model.ema_decay`**: (float) Decay rate for EMA updates of the target encoder (e.g., 0.996).
-   **`jepa_model.target_encoder_mode`**: (string: `"default"`, `"vjepa2"`, `"none"`) Defines the operational mode for target/online encoders (see details in the "Target Encoder Mode" section).
-   **`learning_rate_jepa`**: (float) Learning rate for the JEPA online encoder and predictor.
-   **`batch_size`**: (integer, top-level) Batch size for training.
-   **`num_epochs`**: (integer, top-level) Main number of training epochs for JEPA.
-   **`early_stopping.metric_jepa`**: (string, e.g., `"val_loss_jepa"`) Metric for JEPA early stopping.
-   **`early_stopping.checkpoint_path_jepa`**: (string) Filename for saving the best JEPA model.
-   **`training_options.skip_jepa_training_if_loaded`**: (boolean) If true, skips JEPA training if a model is loaded.

### Auxiliary Loss Parameters (under `auxiliary_loss`):

-   **`auxiliary_loss.type`**: (string, e.g., `"vicreg"`, `"barlow_twins"`, `"dino"`) Type of auxiliary loss.
-   **`auxiliary_loss.weight`**: (float) Weight factor for the auxiliary loss term.
-   **`auxiliary_loss.params.*`**: Nested dictionary for parameters specific to the chosen auxiliary loss (e.g., `auxiliary_loss.params.vicreg.sim_coeff`).

### JEPAStateDecoder Parameters (under `jepa_decoder_training`):

The `JEPAStateDecoder` reuses the general Transformer decoder architecture defined by parameters like `decoder_dim`, `decoder_depth`, `decoder_heads`, `decoder_mlp_dim`, and `decoder_patch_size` (top-level in `config.yaml`).

-   **`jepa_decoder_training.enabled`**: (boolean) Set to `true` to enable training of the JEPA state decoder after JEPA model training.
-   **`jepa_decoder_training.num_epochs`**: (integer) Number of epochs for training the state decoder.
-   **`jepa_decoder_training.learning_rate`**: (float) Learning rate for the state decoder.
-   **`jepa_decoder_training.batch_size`**: (integer) Batch size for decoder training.
-   **`jepa_decoder_training.embedding_source`**: (string: `"online"`, `"target"`) Specifies whether to use embeddings from the JEPA online or target encoder as input for reconstruction.
-   **`jepa_decoder_training.checkpoint_path`**: (string) Filename for saving the best state decoder model.
-   **`jepa_decoder_training.validation_plot_dir`**: (string) Directory to save validation image reconstructions.
-   **`jepa_decoder_training.early_stopping.metric`**: (string, e.g., `"val_loss_jepa_decoder"`) Metric for state decoder early stopping.
-   **`jepa_decoder_training.early_stopping.patience`**: (integer) Patience for state decoder early stopping.
-   **`jepa_decoder_training.early_stopping.delta`**: (float) Minimum change for state decoder early stopping.

For a comprehensive list and detailed explanations, refer to the heavily commented `config.yaml` and **[`docs/06_usage_guide.md`](../06_usage_guide.md)**.
