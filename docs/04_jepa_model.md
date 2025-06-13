# 04 Joint Embedding Predictive Architecture (JEPA) Model

The Joint Embedding Predictive Architecture (JEPA), as implemented in `src/models/jepa.py`, offers an alternative to direct pixel-space prediction by operating entirely within an abstract embedding space. This approach is designed to encourage the learning of more semantically meaningful and efficient representations by abstracting away irrelevant pixel-level details. The core idea is to predict the embedding of a future state `s_t+1` given an embedding of the current state `s_t` and the action `a_t`.

<!-- TODO: Add a high-level architectural diagram for JEPA -->
```
s_t (image) ---+--> [ONLINE ENCODER] ----> online_encoded_s_t (z_t) ------> (to Auxiliary Loss)
               |         ^      |
               |         | EMA Update
               |         |      |
               +--> [TARGET ENCODER] ---> target_encoded_s_t (z'_t) --+
                                                                       |
a_t (vector) --> [ACTION_EMB] ---------> embedded_a_t (a_emb_t) ------+--> [PREDICTOR] --> predicted_embedding_s_t+1 (z_hat_{t+1}) --+
                                                                                                                                    | (Prediction Loss)
s_t+1 (image) --> [ONLINE ENCODER] ----> online_encoded_s_t+1 (z_{t+1}) -> (to Auxiliary Loss)                                         |
                |                                                                                                                     |
                +--> [TARGET ENCODER] ---> target_encoded_s_t+1 (z'_{t+1})-------------------------------------------------------------+
```
*(Simplified ASCII diagram illustrating data flow and main components)*

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

### 4. Target Encoder Mode (`target_encoder_mode`)

The `target_encoder_mode` parameter in the `jepa_model` section of `config.yaml` (and passed to the `JEPA` class constructor) dictates the operational strategy for the online and target encoders, influencing how predictions are made, how targets are generated, and how EMA updates and auxiliary losses are handled. It offers three distinct modes:

-   **`"default"`**:
    -   **Description:** This is the standard JEPA behavior as described in earlier sections.
    -   **Data Flow:**
        -   Current state `s_t` is encoded by the **target encoder**: `target_encoded_s_t = target_encoder(s_t)`.
        -   Next state `s_t_plus_1` is encoded by the **target encoder**: `target_encoded_s_t_plus_1 = target_encoder(s_t_plus_1)`. This serves as the prediction target.
        -   The predictor takes `target_encoded_s_t` and the embedded action to predict `target_encoded_s_t_plus_1`.
    -   **EMA Updates:** The target encoder is updated via EMA of the online encoder's weights once per training step (typically after the optimizer step, managed by `perform_ema_update()` called from the training loop).
    -   **Auxiliary Loss:** Calculated using online encoder representations of both `s_t` (`online_encoded_s_t`) and `s_t_plus_1` (`online_encoded_s_t_plus_1`). Both representations are returned by the `forward` method.

-   **`"vjepa2"`**:
    -   **Description:** This mode is inspired by aspects of the V-JEPA 2 (Video Joint Embedding Predictive Architecture) approach, focusing on predicting the target from an online-encoded current state.
    -   **Data Flow:**
        -   Current state `s_t` is encoded by the **online encoder**: `online_encoded_s_t = online_encoder(s_t)`.
        -   *Crucially, the EMA update for the target encoder is performed at this point, within the `forward` pass, before the target encoder is used for `s_t_plus_1`.*
        -   Next state `s_t_plus_1` is encoded by the **target encoder** (now updated): `target_encoded_s_t_plus_1 = target_encoder(s_t_plus_1)`. This (detached) representation serves as the prediction target.
        -   The predictor takes `online_encoded_s_t` (from the online encoder) and the embedded action to predict `target_encoded_s_t_plus_1`.
    -   **EMA Updates:** The target encoder is updated via EMA within the `forward` method of the JEPA model, specifically after `s_t` is processed by the online encoder and before `s_t_plus_1` is processed by the target encoder. The general `perform_ema_update()` called by the training loop will do nothing in this mode.
    -   **Auxiliary Loss:** Calculated **only** on the online encoder's representation of the current state `s_t` (`online_encoded_s_t`). The `forward` method returns `None` for the `online_encoded_s_t_plus_1` representation to signal this.

-   **`"none"`**:
    -   **Description:** In this mode, the target encoder is entirely disabled. All predictions are based on the online encoder.
    -   **Data Flow:**
        -   Current state `s_t` is encoded by the **online encoder**: `online_encoded_s_t = online_encoder(s_t)`.
        -   Next state `s_t_plus_1` is also encoded by the **online encoder**: `online_encoded_s_t_plus_1 = online_encoder(s_t_plus_1)`. The (detached) version of this serves as the prediction target.
        -   The predictor takes `online_encoded_s_t` and the embedded action to predict `online_encoded_s_t_plus_1.detach()`.
    -   **EMA Updates:** No EMA updates are performed as there is effectively no target encoder to update. The `perform_ema_update()` method does nothing.
    -   **Auxiliary Loss:** Calculated using online encoder representations of both `s_t` (`online_encoded_s_t`) and `s_t_plus_1` (`online_encoded_s_t_plus_1`). Both representations are returned by the `forward` method.

Choosing the appropriate `target_encoder_mode` allows for flexibility in experimenting with different JEPA-style architectures and update rules, potentially impacting representation quality and model performance.

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
