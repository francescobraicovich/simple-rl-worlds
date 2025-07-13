# Training Scripts and Workflows

This document explains the purpose and operation of each training script. These scripts orchestrate the training of the models defined in `01_models.md`.

---

## 1. `collect_load_data.py`

*   **Goal:** To create a dataset of experiences by having an agent interact with the environment.

*   **Workflow:**
    1.  **PPO Agent:** It uses a Proximal Policy Optimization (PPO) agent from the `stable-baselines3` library to play the game.
    2.  **Data Collection:** The agent runs for a specified number of episodes, and for each step, it records the `(state, action, reward, next_state)` tuple.
    3.  **Dataset Creation:** The collected experiences are saved to a file (e.g., `assault_rep_4.pth`). This script contains two main classes:
        *   `DataCollectionPipeline`: If no dataset file is found, it trains a PPO agent from scratch and generates the data.
        *   `DataLoadingPipeline`: Assumes the data already exists and simply loads it.
    4.  **DataLoaders:** The script creates PyTorch `DataLoader` objects for the training and validation splits, which are used by all other training scripts.

*   **How to Run:** This script is typically run once at the beginning of an experiment.
    ```bash
    python src/scripts/collect_load_data.py
    ```

---

## 2. `train_encoder_decoder.py`

*   **Goal:** To train the `Encoder`, `Predictor`, and `Decoder` models jointly in an end-to-end fashion.

*   **Workflow:** This script represents the classic world model approach.
    1.  **Input:** A batch of `(state, action, next_state)` data.
    2.  **Encode:** The `state` is passed through the `Encoder` to get a latent vector `z_t`.
    3.  **Predict:** The latent vector `z_t` and the `action` are passed to the `Predictor` to get a predicted next latent state, `z_hat_{t+1}`.
    4.  **Decode:** The predicted latent state `z_hat_{t+1}` is passed to the `Decoder` to reconstruct the next state image, `s_hat_{t+1}`.
    5.  **Loss Calculation:** The loss is the Mean Absolute Error (L1 Loss) between the reconstructed image `s_hat_{t+1}` and the real `next_state` image.
    6.  **Backpropagation:** The loss is backpropagated through all three models (`Decoder`, `Predictor`, and `Encoder`), updating all of their weights simultaneously.

*   **Inputs:** A dataset created by `collect_load_data.py`.
*   **Outputs:** Trained weights for the encoder, predictor, and decoder, saved in `weights/encoder_decoder/`.

---

## 3. `train_jepa.py`

*   **Goal:** To train the `Encoder` and `Predictor` using the self-supervised Joint-Embedding Predictive Architecture (JEPA) method.

*   **Workflow:** This is the core of the JEPA approach and does *not* use the `Decoder`.
    1.  **Models:** It uses two encoders: an `encoder` (the one being trained) and a `target_encoder` (a slow-moving average of the `encoder`).
    2.  **Prediction in Latent Space:**
        a. The `state` is passed through the main `encoder` to get `z_t`.
        b. `z_t` and the `action` are passed to the `Predictor` to get a predicted next latent state, `z_hat_{t+1}`.
    3.  **Target Generation:**
        a. The *real* `next_state` is passed through the frozen `target_encoder` to get the target latent state, `z_target_{t+1}`.
    4.  **Loss Calculation:** The loss is the Mean Absolute Error (L1 Loss) between the *predicted* latent state `z_hat_{t+1}` and the *target* latent state `z_target_{t+1}`.
    5.  **EMA Update:** After each training step, the weights of the `target_encoder` are updated to be a bit closer to the main `encoder`'s weights (Exponential Moving Average). This provides a stable, slowly evolving target for the predictor to chase.

*   **Inputs:** A dataset from `collect_load_data.py`.
*   **Outputs:** Trained weights for the encoder and predictor, saved in `weights/jepa/`.

---

## 4. `train_jepa_decoder.py`

*   **Goal:** To evaluate the quality of the representations learned by `train_jepa.py`. This is done by training *only* a `Decoder` to see how well it can reconstruct images from the JEPA-trained latent space.

*   **Workflow:**
    1.  **Frozen Models:** The script loads the pre-trained `Encoder` and `Predictor` from the JEPA training and **freezes their weights**.
    2.  **Latent Generation:** It performs the same encoding and prediction steps as `train_jepa.py` but within a `torch.no_grad()` context, meaning no gradients are computed for the encoder and predictor.
    3.  **Decode:** The predicted latent state `z_hat_{t+1}` is passed to the `Decoder`.
    4.  **Loss Calculation:** The reconstruction loss (L1) is calculated between the decoded image and the real `next_state`.
    5.  **Backpropagation:** The loss is backpropagated **only through the `Decoder`**. The Encoder and Predictor are not updated.

*   **Inputs:** A dataset and the pre-trained JEPA `encoder` and `predictor` weights.
*   **Outputs:** A trained `Decoder` whose weights are saved in `weights/jepa_decoder/`.

---

## 5. `train_reward_predictor.py`

*   **Goal:** To train a `RewardPredictor` on top of a frozen, pre-trained `Encoder`.

*   **Workflow:**
    1.  **Frozen Encoder:** Loads a pre-trained `Encoder` (either from the JEPA or Encoder-Decoder training) and freezes its weights.
    2.  **Latent Generation:** The `state` and `next_state` are both passed through the frozen `Encoder` to get their latent representations, `z_t` and `z_{t+1}`.
    3.  **Reward Prediction:** Both `z_t` and `z_{t+1}` are passed to the `RewardPredictor`.
    4.  **Loss Calculation:** The loss is the Mean Squared Error (MSE) between the predicted reward and the real `reward` from the dataset.
    5.  **Backpropagation:** The loss is backpropagated only through the `RewardPredictor`.

*   **Inputs:** A dataset and a pre-trained `Encoder`.
*   **Outputs:** A trained `RewardPredictor`, saved in a subdirectory corresponding to the encoder used (e.g., `weights/jepa/reward_predictor/`).

---

## 6. `train_dynamics_reward_predictor.py`

*   **Goal:** To test how well the learned dynamics model can be used for reward prediction.

*   **Workflow:** This is a crucial test of the world model. Instead of giving the reward predictor the *true* next latent state, we give it the *predicted* one.
    1.  **Frozen Models:** Loads and freezes a pre-trained `Encoder` and `Predictor`.
    2.  **Latent Generation & Prediction:**
        a. The `state` is passed through the `Encoder` to get `z_t`.
        b. `z_t` and the `action` are passed to the `Predictor` to get the *predicted* next latent state, `z_hat_{t+1}`.
    3.  **Reward Prediction:** `z_t` and the **predicted** `z_hat_{t+1}` are passed to the `RewardPredictor`.
    4.  **Loss & Backpropagation:** The MSE loss is calculated against the real reward and backpropagated only through the `RewardPredictor`.

*   **Inputs:** A dataset and a pre-trained `Encoder` and `Predictor`.
*   **Outputs:** A trained `RewardPredictor`, saved in a subdirectory (e.g., `weights/jepa/dynamics_reward_predictor/`).
